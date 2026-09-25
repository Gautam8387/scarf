"""Resumable CELLxGENE downloads with verified bucket chunk checkpoints."""

import hashlib
import logging
import os
import re
from collections.abc import Callable
from concurrent.futures import CancelledError, ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Lock, Thread
from time import monotonic
from uuid import UUID

import httpx
from pydantic import BaseModel, ConfigDict, Field

from .._storage import Bucket, dataset_prefix, error_message, retry
from .models import DatasetRecord

DOWNLOADS_PREFIX = "_internal/downloads"
CHUNK_BYTES = 512 * 1024 * 1024
_BUFFER_BYTES = 1024 * 1024


def download_connections() -> int:
    """Return the bounded number of concurrent CELLxGENE range requests."""
    try:
        connections = int(os.environ.get("CYTEBASE_DOWNLOAD_CONNECTIONS", "2"))
    except ValueError as error:
        raise ValueError("CYTEBASE_DOWNLOAD_CONNECTIONS must be 1 to 4") from error
    if not 1 <= connections <= 4:
        raise ValueError("CYTEBASE_DOWNLOAD_CONNECTIONS must be 1 to 4")
    return connections


class _DownloadProgress:
    """Aggregate source activity without allowing workers to change the stage."""

    def __init__(self, progress: Callable | None, storage: Bucket, total: int | None):
        self.progress, self.storage = progress, storage
        self.lock, self.stopped = Lock(), Event()
        self.fetched: dict[int, int] = {}
        self.active: set[int] = set()
        self.retries: dict[int, dict] = {}
        self.resume_bytes = 0
        self.stage = "downloading"
        self.values = {
            "completed": 0,
            "total": total,
            "unit": "bytes",
            "checkpointBytes": 0,
            "verifiedBytes": 0,
            "checkpointPaths": [],
            "consumerStep": "probe",
        }

    def _emit(self) -> None:
        # Serialize emissions as well as snapshots so an older stage cannot win.
        with self.lock:
            if self.progress is not None:
                self.progress(
                    self.stage,
                    **self.values,
                    fetchedBytes=self.resume_bytes + sum(self.fetched.values()),
                    activeFetches=len(self.active),
                    sourceRetries=[self.retries[key] for key in sorted(self.retries)],
                )

    def update(self, stage: str | None = None, **values) -> None:
        with self.lock:
            if stage is not None:
                self.stage = stage
            self.values.update(values)
        self._emit()

    def source_bytes(self, start: int, size: int) -> None:
        with self.lock:
            # Count each byte range once even if a source retry restarts its file.
            self.fetched[start] = max(self.fetched.get(start, 0), size)

    def source_retry(self, start: int, stage: str, **values) -> None:
        with self.lock:
            self.retries[start] = {"start": start, "stage": stage, **values}

    def heartbeat(self) -> None:
        # Refresh local counters; the Modal callback retains its 15-second flush.
        while not self.stopped.wait(1):
            try:
                self._emit()
            except Exception as error:
                logging.warning(
                    "Download progress update failed: %s", error_message(error)
                )

    def __enter__(self):
        self.previous_progress = self.storage.progress
        self.storage.progress = self.update
        self.thread = Thread(target=self.heartbeat, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stopped.set()
        self.thread.join()
        self.storage.progress = self.previous_progress


class _Chunk(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    start: int = Field(ge=0)
    end: int = Field(ge=0)
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class _Checkpoint(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    datasetId: str
    datasetVersionId: str
    sourceUrl: str
    etag: str
    totalBytes: int = Field(gt=0)
    chunkBytes: int = CHUNK_BYTES
    chunks: list[_Chunk] = Field(default_factory=list)
    createdAt: str | None = None
    updatedAt: str | None = None


class _Completion(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    datasetId: str
    datasetVersionId: str
    sourceUrl: str
    etag: str
    sourceBytes: int = Field(gt=0)
    sourceSha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    completedAt: str


def checkpoint_prefix(dataset_id: str, version_id: str) -> str:
    """Keep checkpoint paths confined to a validated dataset and version."""
    return f"{DOWNLOADS_PREFIX}/{UUID(str(dataset_id))}/{UUID(str(version_id))}"


def list_checkpoint_paths(
    dataset_id: str, version_id: str, storage: Bucket
) -> list[str]:
    """Return exact staging paths, including chunks absent from the manifest."""
    return storage.list_files(f"{checkpoint_prefix(dataset_id, version_id)}/")


def inventory_downloads(storage: Bucket) -> list[dict]:
    """Inspect receipts and committed ranges without downloading chunk contents."""
    grouped: dict[tuple[str, str], set[str]] = {}
    root = f"{DOWNLOADS_PREFIX}/"
    for path in storage.list_files(root):
        parts = path.removeprefix(root).split("/")
        key = (parts[0], parts[1] if len(parts) > 1 else "")
        grouped.setdefault(key, set()).add(path)
    rows = []
    for (dataset_id, version_id), paths in sorted(grouped.items()):
        row = {
            "datasetId": dataset_id,
            "datasetVersionId": version_id,
            "complete": False,
            "committedBytes": 0,
            "totalBytes": None,
            "createdAt": None,
            "updatedAt": None,
            "completedAt": None,
            "paths": sorted(paths),
            "orphanPaths": sorted(paths),
            "errors": [],
        }
        try:
            prefix = checkpoint_prefix(dataset_id, version_id)
            raw = storage.read_json(f"{prefix}/manifest.json")
            if raw is None:
                row["errors"].append("Checkpoint manifest is missing")
            else:
                checkpoint = _Checkpoint.model_validate(raw)
                committed = _validate_checkpoint(
                    checkpoint,
                    prefix,
                    dataset_id,
                    version_id,
                    checkpoint.sourceUrl,
                    checkpoint.totalBytes,
                    checkpoint.etag,
                )
                row.update(
                    committedBytes=committed,
                    totalBytes=checkpoint.totalBytes,
                    createdAt=checkpoint.createdAt,
                    updatedAt=checkpoint.updatedAt,
                )
                known = {
                    f"{prefix}/manifest.json",
                    f"{prefix}/completion.json",
                    *(chunk.path for chunk in checkpoint.chunks),
                }
                row["orphanPaths"] = sorted(paths - known)
                row["errors"].extend(
                    f"Missing committed chunk: {chunk.path}"
                    for chunk in checkpoint.chunks
                    if chunk.path not in paths
                )
                raw_completion = storage.read_json(f"{prefix}/completion.json")
                if raw_completion is not None:
                    completion = _Completion.model_validate(raw_completion)
                    _validate_completion(
                        checkpoint,
                        completion,
                        dataset_id=dataset_id,
                        version_id=version_id,
                        url=checkpoint.sourceUrl,
                        expected_bytes=checkpoint.totalBytes,
                    )
                    row.update(
                        complete=not row["errors"], completedAt=completion.completedAt
                    )
        except (ValueError, KeyError, TypeError) as error:
            row["errors"].append(error_message(error))
        rows.append(row)
    return rows


def downloads(
    storage: Bucket, records: list[DatasetRecord], active_run: dict | None = None
) -> list[dict]:
    """Join staging with known ownership; unresolved work blocks manual pruning."""
    by_id = {str(record.datasetId): record for record in records}
    if len(by_id) != len(records):
        raise ValueError("Dataset ownership is ambiguous: duplicate dataset IDs")
    active = active_run is not None and active_run.get("state") not in {
        "completed",
        "failed",
        "reset",
    }
    operation = (
        {key: active_run.get(key) for key in ("runId", "callId", "action", "state")}
        if active
        else None
    )
    rows = []
    for item in inventory_downloads(storage):
        record = by_id.get(item["datasetId"])
        known = record is not None and item["datasetVersionId"] in {
            str(version.datasetVersionId) for version in record.versions
        } | {str(record.latestVersionId)}
        unresolved = record is not None and record.status == "processing"
        blocked = (
            "Dataset/version ownership cannot be established"
            if not known
            else "Pipeline work is active or unresolved"
            if active or unresolved
            else None
        )
        rows.append(
            item
            | {
                "cytebaseId": record.cytebaseId if record else None,
                "ownershipEstablished": known and not unresolved,
                "activeOperation": operation,
                "prunable": blocked is None,
                "blockedReason": blocked,
            }
        )
    return rows


def prune_groups(
    paths: list[str],
    storage: Bucket,
    records: list[DatasetRecord],
    active_run: dict | None = None,
) -> dict[str, dict]:
    """Require a reviewed full inventory before selecting versions for pruning."""
    if not paths:
        raise ValueError("Deletion requires explicitly reviewed approvedPaths")
    requested = set(paths)
    groups: dict[str, dict] = {}
    matched = set()
    for item in downloads(storage, records, active_run):
        exact = set(item["paths"])
        if not requested.intersection(exact):
            continue
        if not item["prunable"]:
            raise ValueError(item["blockedReason"])
        if not exact.issubset(requested):
            raise ValueError(
                "Review every staging path, including markers and orphan chunks"
            )
        groups.setdefault(item["cytebaseId"], {"pruneVersions": []})[
            "pruneVersions"
        ].append({"datasetVersionId": item["datasetVersionId"], "paths": item["paths"]})
        matched.update(exact)
    if matched != requested:
        raise ValueError(
            "Staging inventory changed or approvedPaths contains unknown paths"
        )
    return groups


def validate_prune_owner(record: DatasetRecord, request: dict) -> None:
    versions = {str(row.datasetVersionId) for row in record.versions} | {
        str(record.latestVersionId)
    }
    if not request.get("pruneVersions"):
        raise ValueError("Prune has no reviewed version inventory")
    for item in request["pruneVersions"]:
        if item["datasetVersionId"] not in versions:
            raise ValueError("Staging version ownership cannot be established")


def run_download(
    record: DatasetRecord,
    request: dict,
    storage: Bucket,
    progress: Callable,
    assert_owner: Callable[[], None],
) -> dict:
    """Populate verified staging; successful existing receipts are reusable."""
    ready = (
        record.status == "ready"
        and record.processedVersionId == record.latestVersionId
        and record.zarrUri is not None
    )
    if ready and request.get("operation") == "process" and not request.get("force"):
        return {
            "outcome": "skipped",
            "message": "The registered version is already ready",
        }
    completion = get_completion(record, storage)
    outcome = "skipped"
    if completion is None:
        # Invalidate a previous partial cleanup before creating replacement chunks.
        record.cleanupIntent = None
        assert_owner()
        storage.write_json(
            f"{dataset_prefix(record.cytebaseId)}/dataset.json",
            record.model_dump(mode="json"),
        )
        with TemporaryDirectory(prefix="cytebase-download-") as directory:
            size, checksum = download_h5ad(
                record.sourceUrl,
                Path(directory) / "source.h5ad",
                record.sourceBytes,
                dataset_id=str(record.datasetId),
                version_id=str(record.latestVersionId),
                storage=storage,
                progress=progress,
                assert_owner=assert_owner,
                timings=record.timings,
            )
        completion = get_completion(record, storage)
        if completion is None or (
            completion["sourceBytes"],
            completion["sourceSha256"],
        ) != (size, checksum):
            raise ValueError("Download receipt differs from verified staged bytes")
        outcome = "succeeded"
    record.sourceBytes = completion["sourceBytes"]
    for version in record.versions:
        if version.datasetVersionId == record.latestVersionId:
            version.sourceSha256 = completion["sourceSha256"]
    record.checkpointPaths = list_checkpoint_paths(
        str(record.datasetId), str(record.latestVersionId), storage
    )
    if not ready:
        record.status = "downloaded"
    return {"outcome": outcome}


def _response_identity(
    response: httpx.Response,
    start: int,
    end: int,
    total: int | None = None,
    etag: str | None = None,
) -> tuple[int, str]:
    response.raise_for_status()
    if response.status_code != 206:
        raise ValueError(
            "CELLxGENE did not honor the byte range or its source changed "
            f"(HTTP {response.status_code}); existing checkpoints were retained"
        )
    match = re.fullmatch(
        r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", "")
    )
    if match is None:
        raise ValueError("CELLxGENE response has no valid Content-Range")
    range_start, range_end, range_total = map(int, match.groups())
    if (
        (range_start, range_end) != (start, end)
        or range_total <= end
        or (total is not None and range_total != total)
    ):
        raise ValueError("CELLxGENE byte range or total size changed")
    response_etag = response.headers.get("ETag", "")
    if not re.fullmatch(r'"[^"\r\n]+"', response_etag):
        raise ValueError("CELLxGENE must supply a strong ETag for safe resume")
    if etag is not None and response_etag != etag:
        raise ValueError("CELLxGENE source ETag changed; checkpoints were retained")
    if response.headers.get("Content-Encoding", "identity").lower() != "identity":
        raise ValueError("Byte-range download requires an uncompressed response")
    length = response.headers.get("Content-Length")
    if length is not None and int(length) != end - start + 1:
        raise ValueError("CELLxGENE range Content-Length does not match its bounds")
    return range_total, response_etag


def _probe(client: httpx.Client, url: str) -> tuple[int, str]:
    with client.stream("GET", url, headers={"Range": "bytes=0-0"}) as response:
        identity = _response_identity(response, 0, 0)
        size = 0
        for block in response.iter_raw(chunk_size=_BUFFER_BYTES):
            size += len(block)
            if size > 1:
                raise ValueError("CELLxGENE probe returned more bytes than requested")
        if size != 1:
            raise httpx.RemoteProtocolError("CELLxGENE probe returned incomplete data")
        return identity


def _chunk_path(prefix: str, start: int, end: int) -> str:
    return f"{prefix}/chunks/{start:020d}-{end:020d}.bin"


def _validate_checkpoint(
    checkpoint: _Checkpoint,
    prefix: str,
    dataset_id: str,
    version_id: str,
    url: str,
    total: int,
    etag: str,
) -> int:
    if (
        checkpoint.datasetId != str(UUID(str(dataset_id)))
        or checkpoint.datasetVersionId != str(UUID(str(version_id)))
        or checkpoint.sourceUrl != url
        or checkpoint.totalBytes != total
        or checkpoint.etag != etag
        or checkpoint.chunkBytes != CHUNK_BYTES
    ):
        raise ValueError(
            "Download checkpoint does not match the current source identity or "
            "chunk size; it was retained for explicit review"
        )
    offset = 0
    for chunk in checkpoint.chunks:
        expected_end = min(offset + CHUNK_BYTES, total) - 1
        if (
            offset >= total
            or chunk.start != offset
            or chunk.end != expected_end
            or chunk.path != _chunk_path(prefix, offset, expected_end)
        ):
            raise ValueError("Download checkpoint has invalid or noncontiguous ranges")
        offset = chunk.end + 1
    return offset


def _validate_completion(
    checkpoint: _Checkpoint,
    completion: _Completion,
    *,
    dataset_id: str,
    version_id: str,
    url: str,
    expected_bytes: int | None,
) -> None:
    prefix = checkpoint_prefix(dataset_id, version_id)
    if (
        completion.datasetId != str(UUID(str(dataset_id)))
        or completion.datasetVersionId != str(UUID(str(version_id)))
        or completion.sourceUrl != url
        or (expected_bytes is not None and completion.sourceBytes != expected_bytes)
        or not re.fullmatch(r'"[^"\r\n]+"', completion.etag)
    ):
        raise ValueError("Completion receipt does not match the registered source")
    completed = datetime.fromisoformat(completion.completedAt)
    if completed.tzinfo is None:
        raise ValueError("Completion receipt must include a timezone")
    committed = _validate_checkpoint(
        checkpoint,
        prefix,
        dataset_id,
        version_id,
        url,
        completion.sourceBytes,
        completion.etag,
    )
    if committed != completion.sourceBytes:
        raise ValueError("Completion receipt refers to an incomplete checkpoint")


def _read_completion(
    dataset_id: str,
    version_id: str,
    url: str,
    expected_bytes: int | None,
    storage: Bucket,
) -> tuple[_Checkpoint, _Completion] | None:
    prefix = checkpoint_prefix(dataset_id, version_id)
    raw = storage.read_json(f"{prefix}/completion.json")
    if raw is None:
        return None
    completion = _Completion.model_validate(raw)
    raw_checkpoint = storage.read_json(f"{prefix}/manifest.json")
    if raw_checkpoint is None:
        raise ValueError(
            "Completion receipt has no checkpoint manifest; staging retained"
        )
    checkpoint = _Checkpoint.model_validate(raw_checkpoint)
    _validate_completion(
        checkpoint,
        completion,
        dataset_id=dataset_id,
        version_id=version_id,
        url=url,
        expected_bytes=expected_bytes,
    )
    return checkpoint, completion


def get_completion(record: DatasetRecord, storage: Bucket) -> dict | None:
    """Read a committed source receipt and validate its checkpoint identity."""
    saved = _read_completion(
        str(record.datasetId),
        str(record.latestVersionId),
        record.sourceUrl,
        record.sourceBytes,
        storage,
    )
    return None if saved is None else saved[1].model_dump()


def _append_verified_chunk(
    chunk: _Chunk,
    source_path: Path,
    output,
    full_digest,
    report: Callable[[int], None],
) -> int:
    """Append one bounded chunk while validating it and hashing the full source."""
    expected_bytes = chunk.end - chunk.start + 1
    if source_path.stat().st_size != expected_bytes:
        raise ValueError(f"Staged chunk size mismatch: {chunk.path}")
    digest = hashlib.sha256()
    copied = 0
    with source_path.open("rb") as source:
        while block := source.read(_BUFFER_BYTES):
            copied += len(block)
            if copied > expected_bytes:
                raise ValueError(f"Staged chunk size mismatch: {chunk.path}")
            digest.update(block)
            full_digest.update(block)
            output.write(block)
            report(chunk.start + copied)
    if copied != expected_bytes:
        raise ValueError(f"Staged chunk size mismatch: {chunk.path}")
    if digest.hexdigest() != chunk.sha256:
        raise ValueError(f"Staged chunk SHA-256 mismatch: {chunk.path}")
    return chunk.start + copied


def _restore_completed(
    checkpoint: _Checkpoint,
    completion: _Completion,
    destination: Path,
    progress: Callable | None,
    storage: Bucket,
) -> tuple[int, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.part")
    prefix = checkpoint_prefix(completion.datasetId, completion.datasetVersionId)
    paths = [
        f"{prefix}/manifest.json",
        f"{prefix}/completion.json",
        *(chunk.path for chunk in checkpoint.chunks),
    ]
    digest = hashlib.sha256()
    assembled = 0

    def report(completed: int, message: str) -> None:
        if progress is not None:
            progress(
                "restoring_download",
                completed=completed,
                total=completion.sourceBytes,
                unit="bytes",
                checkpointBytes=completion.sourceBytes,
                checkpointPaths=paths,
                message=message,
            )

    report(0, "Restoring staged chunks without contacting CELLxGENE")
    try:
        with (
            TemporaryDirectory(
                prefix="cellxgene-restore-", dir=destination.parent
            ) as directory,
            partial.open("wb") as output,
        ):
            chunk_file = Path(directory) / "chunk"
            for chunk in checkpoint.chunks:
                storage.download(chunk.path, chunk_file)
                assembled = _append_verified_chunk(
                    chunk,
                    chunk_file,
                    output,
                    digest,
                    lambda completed: report(
                        completed, "Verifying staged chunk and full source checksums"
                    ),
                )
            if (
                assembled != completion.sourceBytes
                or digest.hexdigest() != completion.sourceSha256
            ):
                raise ValueError("Staged source does not match its completion receipt")
        partial.replace(destination)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    report(assembled, "Staged source restored and verified")
    return assembled, digest.hexdigest()


def restore_h5ad(
    record: DatasetRecord,
    destination: Path,
    storage: Bucket,
    *,
    progress: Callable | None = None,
) -> tuple[int, str]:
    """Verify and reconstruct a completed staged H5AD using HF access only."""
    saved = _read_completion(
        str(record.datasetId),
        str(record.latestVersionId),
        record.sourceUrl,
        record.sourceBytes,
        storage,
    )
    if saved is None:
        raise ValueError(
            "Download prerequisite is incomplete; run /datasets/download first"
        )
    return _restore_completed(*saved, destination, progress, storage)


def _fetch_chunk(
    client: httpx.Client,
    url: str,
    path: Path,
    start: int,
    end: int,
    total: int,
    etag: str,
    report: Callable,
    stopped: Event,
) -> str:
    if stopped.is_set():
        raise CancelledError("Download prefetch was cancelled")
    digest = hashlib.sha256()
    size = 0
    with client.stream(
        "GET",
        url,
        headers={"Range": f"bytes={start}-{end}", "If-Range": etag},
    ) as response:
        _response_identity(response, start, end, total, etag)
        with path.open("wb") as output:
            for block in response.iter_raw(chunk_size=_BUFFER_BYTES):
                if stopped.is_set():
                    raise CancelledError("Download prefetch was cancelled")
                size += len(block)
                if size > end - start + 1:
                    raise ValueError("CELLxGENE returned more bytes than requested")
                output.write(block)
                digest.update(block)
                report(size)
    if size != end - start + 1:
        raise httpx.RemoteProtocolError("CELLxGENE returned an incomplete byte range")
    return digest.hexdigest()


def download_h5ad(
    url: str,
    destination: Path,
    expected_bytes: int | None = None,
    *,
    dataset_id: str,
    version_id: str,
    storage: Bucket,
    progress: Callable | None = None,
    assert_owner: Callable | None = None,
    timings: dict[str, float] | None = None,
) -> tuple[int, str]:
    """Prefetch bounded ranges and commit verified HF chunks in byte order.

    Local prefetched files never advance the checkpoint. Upload and manifest
    writes remain separate because HF operations are not transactional. A final
    completion receipt covers the full source and every read-back bucket chunk.
    """
    connections = download_connections()
    timings = timings if timings is not None else {}
    timings["downloadSourceWaitSeconds"] = 0.0
    timings["downloadCheckpointSeconds"] = 0.0
    prefix = checkpoint_prefix(dataset_id, version_id)
    manifest_path = f"{prefix}/manifest.json"
    saved = _read_completion(dataset_id, version_id, url, expected_bytes, storage)
    if saved is not None:
        started = monotonic()
        try:
            return _restore_completed(*saved, destination, progress, storage)
        finally:
            timings["downloadCheckpointSeconds"] += monotonic() - started
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.part")
    checkpoint_paths = list_checkpoint_paths(dataset_id, version_id, storage)

    with (
        _DownloadProgress(progress, storage, expected_bytes) as report,
        httpx.Client(
            timeout=httpx.Timeout(60, connect=30),
            follow_redirects=True,
            headers={"Accept-Encoding": "identity"},
            limits=httpx.Limits(max_connections=connections),
        ) as client,
    ):
        report.update(
            "downloading",
            message="Checking source identity and saved checkpoints",
            checkpointPaths=sorted(set(checkpoint_paths)),
        )
        total, etag = retry(lambda: _probe(client, url), progress=report.update)
        report.update(total=total)
        if expected_bytes is not None and total != expected_bytes:
            raise ValueError(
                f"CELLxGENE source is {total} bytes, expected {expected_bytes}"
            )
        raw = storage.read_json(manifest_path)
        if raw is None:
            checkpoint = _Checkpoint(
                datasetId=str(UUID(str(dataset_id))),
                datasetVersionId=str(UUID(str(version_id))),
                sourceUrl=url,
                etag=etag,
                totalBytes=total,
                createdAt=datetime.now(UTC).isoformat(),
                updatedAt=datetime.now(UTC).isoformat(),
            )
            # The server can commit even when its response never reaches us.
            checkpoint_paths.append(manifest_path)
            report.update(
                "checkpointing_download",
                consumerStep="checkpoint",
                message="Creating the download checkpoint",
                checkpointPaths=sorted(set(checkpoint_paths)),
            )
            if assert_owner is not None:
                assert_owner()
            storage.write_json(manifest_path, checkpoint.model_dump())
        else:
            checkpoint = _Checkpoint.model_validate(raw)
        checkpoint_bytes = _validate_checkpoint(
            checkpoint, prefix, dataset_id, version_id, url, total, etag
        )
        checkpoint_paths.extend([manifest_path, *(c.path for c in checkpoint.chunks)])
        with report.lock:
            report.resume_bytes = checkpoint_bytes
        report.update(
            completed=checkpoint_bytes,
            checkpointBytes=checkpoint_bytes,
            checkpointPaths=sorted(set(checkpoint_paths)),
        )
        full_digest = hashlib.sha256()
        assembled = 0
        stopped = Event()
        source_errors: list[BaseException] = []
        error_lock = Lock()

        def check_sources() -> None:
            with error_lock:
                if source_errors:
                    raise source_errors[0]

        def fetch(start: int, end: int, chunk_file: Path) -> str:
            def attempt() -> str:
                if stopped.is_set():
                    raise CancelledError("Download prefetch was cancelled")
                with report.lock:
                    report.active.add(start)
                    report.retries.pop(start, None)
                try:
                    return _fetch_chunk(
                        client,
                        url,
                        chunk_file,
                        start,
                        end,
                        total,
                        etag,
                        lambda size: report.source_bytes(start, size),
                        stopped,
                    )
                finally:
                    with report.lock:
                        report.active.discard(start)

            try:
                return retry(
                    attempt,
                    progress=lambda stage, **values: report.source_retry(
                        start, stage, **values
                    ),
                    stop_event=stopped,
                )
            except BaseException as error:
                if not isinstance(error, CancelledError):
                    with error_lock:
                        source_errors.append(error)
                stopped.set()
                raise

        try:
            with (
                TemporaryDirectory(
                    prefix="cellxgene-chunk-", dir=destination.parent
                ) as directory,
                partial.open("wb") as output,
            ):
                chunk_dir = Path(directory)
                restored = chunk_dir / "restored"
                for chunk in checkpoint.chunks:
                    report.update(
                        "restoring_download",
                        consumerStep="restore",
                        message="Restoring and verifying committed HF chunks",
                    )
                    started = monotonic()
                    try:
                        storage.download(chunk.path, restored)
                        assembled = _append_verified_chunk(
                            chunk,
                            restored,
                            output,
                            full_digest,
                            lambda completed: report.update(verifiedBytes=completed),
                        )
                    finally:
                        timings["downloadCheckpointSeconds"] += monotonic() - started
                    restored.unlink()

                pending = {}
                next_start = assembled
                executor = ThreadPoolExecutor(
                    max_workers=connections, thread_name_prefix="cytebase-fetch"
                )

                def fill_window() -> None:
                    nonlocal next_start
                    check_sources()
                    # The oldest entry remains in the window until HF verification
                    # and hashing finish, so the consumer also occupies one slot.
                    while len(pending) < 2 * connections and next_start < total:
                        check_sources()
                        end = min(next_start + CHUNK_BYTES, total) - 1
                        chunk_file = chunk_dir / f"{next_start:020d}.bin"
                        pending[next_start] = (
                            executor.submit(fetch, next_start, end, chunk_file),
                            chunk_file,
                            end,
                        )
                        next_start = end + 1

                try:
                    fill_window()
                    while pending:
                        start = assembled
                        future, chunk_file, end = pending[start]
                        report.update(
                            "downloading",
                            consumerStep="waiting_for_source",
                            message="Waiting for the next prefetched byte range",
                        )
                        started = monotonic()
                        try:
                            try:
                                sha256 = future.result()
                            except CancelledError:
                                check_sources()
                                raise
                        finally:
                            timings["downloadSourceWaitSeconds"] += (
                                monotonic() - started
                            )
                        check_sources()
                        chunk = _Chunk(
                            start=start,
                            end=end,
                            path=_chunk_path(prefix, start, end),
                            sha256=sha256,
                        )
                        # Include the exact path if an interrupted upload leaves an orphan.
                        checkpoint_paths.append(chunk.path)
                        report.update(
                            "checkpointing_download",
                            consumerStep="upload",
                            message="Uploading completed chunk",
                            checkpointPaths=sorted(set(checkpoint_paths)),
                        )
                        started = monotonic()
                        try:
                            if assert_owner is not None:
                                assert_owner()
                            storage.upload([(chunk_file, chunk.path)])
                            check_sources()
                            updated = checkpoint.model_copy(
                                update={
                                    "chunks": [*checkpoint.chunks, chunk],
                                    "updatedAt": datetime.now(UTC).isoformat(),
                                }
                            )
                            report.update(
                                "checkpointing_download",
                                consumerStep="checkpoint",
                                message="Committing the uploaded chunk",
                            )
                            if assert_owner is not None:
                                assert_owner()
                            storage.write_json(manifest_path, updated.model_dump())
                            checkpoint = updated
                            checkpoint_bytes = end + 1
                            report.update(
                                "verifying_download",
                                consumerStep="verify",
                                completed=checkpoint_bytes,
                                checkpointBytes=checkpoint_bytes,
                                message="Reading back and verifying the committed chunk",
                            )
                            # Release the source copy before downloading HF's copy;
                            # the consumer continues occupying the same window slot.
                            chunk_file.unlink()
                            storage.download(chunk.path, chunk_file)
                            assembled = _append_verified_chunk(
                                chunk,
                                chunk_file,
                                output,
                                full_digest,
                                lambda completed: report.update(
                                    "verifying_download", verifiedBytes=completed
                                ),
                            )
                        finally:
                            timings["downloadCheckpointSeconds"] += (
                                monotonic() - started
                            )
                        chunk_file.unlink()
                        del pending[start]
                        fill_window()
                finally:
                    stopped.set()
                    for future, _, _ in pending.values():
                        future.cancel()
                    # Stop and join before the temporary files or shared client close.
                    executor.shutdown(wait=True, cancel_futures=True)
                if assembled != total:
                    raise ValueError(f"Assembled {assembled} bytes, expected {total}")
            partial.replace(destination)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
        receipt = _Completion(
            datasetId=str(UUID(str(dataset_id))),
            datasetVersionId=str(UUID(str(version_id))),
            sourceUrl=url,
            etag=etag,
            sourceBytes=assembled,
            sourceSha256=full_digest.hexdigest(),
            completedAt=datetime.now(UTC).isoformat(),
        )
        completion_path = f"{prefix}/completion.json"
        report.update(
            "checkpointing_download",
            consumerStep="completion",
            message="Committing the verified source completion receipt",
        )
        if assert_owner is not None:
            assert_owner()
        storage.write_json(completion_path, receipt.model_dump())
        checkpoint_paths.append(completion_path)
        report.update(
            "downloaded",
            consumerStep="complete",
            completed=assembled,
            verifiedBytes=assembled,
            checkpointPaths=sorted(set(checkpoint_paths)),
            message="Verified staged source and committed completion receipt",
        )
        return assembled, receipt.sourceSha256
