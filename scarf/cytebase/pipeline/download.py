"""Bounded CELLxGENE range downloads assembled into a local H5AD."""

import hashlib
import logging
import os
import re
from collections.abc import Callable
from concurrent.futures import CancelledError, ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Lock, Thread
from time import monotonic

import httpx

from .._storage import error_message, retry

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
    """Aggregate fetch counters without allowing fetchers to change the stage."""

    def __init__(self, progress: Callable | None, total: int | None):
        self.progress = progress
        self.lock, self.stopped = Lock(), Event()
        self.fetched: dict[int, int] = {}
        self.active: set[int] = set()
        self.retries: dict[int, dict] = {}
        self.assembled = 0
        self.stage = "downloading"
        self.values = {"total": total, "unit": "bytes", "consumerStep": "probe"}

    def _emit(self) -> None:
        # Serialize emissions so a heartbeat cannot replace a newer transition.
        with self.lock:
            if self.progress is not None:
                self.progress(
                    self.stage,
                    **self.values,
                    completed=self.assembled,
                    assembledBytes=self.assembled,
                    fetchedBytes=sum(self.fetched.values()),
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
            # Retrying a range never counts the same bytes twice or regresses.
            self.fetched[start] = max(self.fetched.get(start, 0), size)

    def assembled_bytes(self, size: int) -> None:
        with self.lock:
            self.assembled = size

    def source_retry(self, start: int, stage: str, **values) -> None:
        with self.lock:
            self.retries[start] = {"start": start, "stage": stage, **values}

    def heartbeat(self) -> None:
        # The outer Modal progress callback retains its 15-second publication.
        while not self.stopped.wait(1):
            try:
                self._emit()
            except Exception as error:
                logging.warning(
                    "Download progress update failed: %s", error_message(error)
                )

    def __enter__(self):
        self.thread = Thread(target=self.heartbeat, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stopped.set()
        self.thread.join()


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
            f"(HTTP {response.status_code})"
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
        raise ValueError("CELLxGENE must supply a strong ETag for range downloads")
    if etag is not None and response_etag != etag:
        raise ValueError("CELLxGENE source ETag changed during download")
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


def _append_verified_chunk(
    path: Path,
    start: int,
    end: int,
    checksum: str,
    output,
    full_digest,
    report: Callable,
    stopped: Event,
) -> int:
    """Recheck local chunk bytes while assembling and hashing the full source."""
    expected_bytes = end - start + 1
    if path.stat().st_size != expected_bytes:
        raise ValueError(f"Local chunk size mismatch at byte {start}")
    digest = hashlib.sha256()
    copied = 0
    with path.open("rb") as source:
        while block := source.read(_BUFFER_BYTES):
            if stopped.is_set():
                raise CancelledError("Download prefetch was cancelled")
            copied += len(block)
            if copied > expected_bytes:
                raise ValueError(f"Local chunk size mismatch at byte {start}")
            digest.update(block)
            full_digest.update(block)
            output.write(block)
            report(start + copied)
    if copied != expected_bytes or digest.hexdigest() != checksum:
        raise ValueError(f"Local chunk checksum or size mismatch at byte {start}")
    return start + copied


def download_h5ad(
    url: str,
    destination: Path,
    expected_bytes: int | None = None,
    *,
    progress: Callable | None = None,
    timings: dict[str, float] | None = None,
) -> tuple[int, str]:
    """Prefetch bounded ranges, verify and assemble locally, and return size/hash.

    No download state is persisted remotely. Failure removes this call's partial
    files, and a later invocation starts from the source again. The destination
    and its adjacent ``.part`` path must both be absent.
    """
    connections = download_connections()
    destination = Path(destination)
    partial = destination.with_name(f"{destination.name}.part")
    if os.path.lexists(destination) or os.path.lexists(partial):
        raise FileExistsError("Download destination and .part path must be absent")
    destination.parent.mkdir(parents=True, exist_ok=True)
    timings = timings if timings is not None else {}
    timings["downloadSourceWaitSeconds"] = 0.0
    timings["downloadAssembleSeconds"] = 0.0

    with (
        _DownloadProgress(progress, expected_bytes) as report,
        httpx.Client(
            timeout=httpx.Timeout(60, connect=30),
            follow_redirects=True,
            headers={"Accept-Encoding": "identity"},
            limits=httpx.Limits(max_connections=connections),
        ) as client,
    ):
        report.update(message="Checking source identity and byte-range support")
        started = monotonic()
        try:
            total, etag = retry(
                lambda: _probe(client, url),
                progress=lambda stage, **values: report.source_retry(
                    -1, stage, **values
                ),
            )
        finally:
            timings["downloadSourceWaitSeconds"] += monotonic() - started
        if expected_bytes is not None and total != expected_bytes:
            raise ValueError(
                f"CELLxGENE source is {total} bytes, expected {expected_bytes}"
            )
        with report.lock:
            report.retries.pop(-1, None)
        report.update(total=total)
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

        # Exclusive creation protects a pre-existing partial file even if it was
        # created after the initial check. Only our own partial is cleaned up.
        output = partial.open("xb")
        try:
            with (
                output,
                TemporaryDirectory(
                    prefix="cellxgene-chunks-", dir=destination.parent
                ) as directory,
            ):
                chunk_dir = Path(directory)
                pending = {}
                next_start = 0
                executor = ThreadPoolExecutor(
                    max_workers=connections, thread_name_prefix="cytebase-fetch"
                )

                def fill_window() -> None:
                    nonlocal next_start
                    check_sources()
                    # The consumer keeps its slot until verification and removal.
                    while len(pending) < 2 * connections and next_start < total:
                        check_sources()
                        end = min(next_start + CHUNK_BYTES, total) - 1
                        path = chunk_dir / f"{next_start:020d}.bin"
                        pending[next_start] = (
                            executor.submit(fetch, next_start, end, path),
                            path,
                            end,
                        )
                        next_start = end + 1

                try:
                    fill_window()
                    while pending:
                        start = assembled
                        future, path, end = pending[start]
                        report.update(
                            consumerStep="waiting_for_source",
                            message="Waiting for the next prefetched byte range",
                        )
                        started = monotonic()
                        try:
                            checksum = future.result()
                        finally:
                            timings["downloadSourceWaitSeconds"] += (
                                monotonic() - started
                            )
                        check_sources()
                        report.update(
                            consumerStep="assemble",
                            message="Verifying and assembling a local chunk",
                        )
                        started = monotonic()
                        try:
                            assembled = _append_verified_chunk(
                                path,
                                start,
                                end,
                                checksum,
                                output,
                                full_digest,
                                report.assembled_bytes,
                                stopped,
                            )
                            path.unlink()
                        finally:
                            timings["downloadAssembleSeconds"] += monotonic() - started
                        del pending[start]
                        fill_window()
                except CancelledError:
                    check_sources()
                    raise
                finally:
                    stopped.set()
                    for future, _, _ in pending.values():
                        future.cancel()
                    # Join before temporary files or the shared client are closed.
                    executor.shutdown(wait=True, cancel_futures=True)
                if assembled != total:
                    raise ValueError(f"Assembled {assembled} bytes, expected {total}")
            if partial.stat().st_size != total:
                raise ValueError("Assembled H5AD size differs from its source")
            # Publish the completed inode without ever replacing an existing path.
            # Both paths share a filesystem; this also avoids copying the H5AD.
            destination.hardlink_to(partial)
        finally:
            partial.unlink(missing_ok=True)
        report.update(
            "downloaded",
            consumerStep="complete",
            message="Local H5AD assembled and verified",
        )
        return assembled, full_digest.hexdigest()
