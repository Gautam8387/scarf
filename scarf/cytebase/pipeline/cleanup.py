"""Delete staged bytes only after verified publication or exact manual approval."""

import logging
from collections.abc import Callable
from datetime import UTC, datetime

from .._storage import Bucket, dataset_prefix, error_message, retry
from .download import (
    checkpoint_prefix,
    get_completion,
    list_checkpoint_paths,
    validate_prune_owner,
)
from .models import DatasetRecord


def remove_staging(
    dataset_id: str,
    version_id: str,
    storage: Bucket,
    *,
    exact_paths: list[str],
    progress: Callable,
    assert_owner: Callable[[], None],
) -> list[str]:
    """Remove markers before precisely inventoried chunks, then confirm removal."""
    prefix = f"{checkpoint_prefix(dataset_id, version_id)}/"
    approved = set(exact_paths)
    for path in approved:
        if (
            not path.startswith(prefix)
            or any(part in {"", ".", ".."} for part in path.split("/"))
            or any(character in path for character in "*?[]\\")
        ):
            raise ValueError(f"Unsafe staging path: {path}")

    def inventory() -> list[str]:
        return list_checkpoint_paths(dataset_id, version_id, storage)

    paths = inventory()
    if set(paths) != approved:
        raise ValueError("Staging inventory changed; review its exact paths again")

    def remove(selected: list[str]) -> None:
        current = set(inventory())
        if current - approved:
            raise ValueError("Staging inventory changed during removal; files retained")
        pending = sorted(set(selected) & current)
        if pending:
            assert_owner()
            storage.delete_exact(pending)

    markers = [f"{prefix}completion.json", f"{prefix}manifest.json"]
    for marker in markers:
        if marker in paths:
            remove([marker])
            if marker in inventory():
                raise RuntimeError(
                    f"Staging marker removal was not confirmed: {marker}"
                )
    chunks = [path for path in paths if path not in markers]
    if chunks:
        progress("cleaning_downloads", message="Removing verified staged source chunks")
        remove(chunks)
    remaining = inventory()
    progress(
        "cleaning_downloads",
        checkpointPaths=remaining,
        message="Checked staging for remaining files",
    )
    return remaining


def _cleanup(
    record: DatasetRecord,
    storage: Bucket,
    progress: Callable,
    assert_owner: Callable[[], None],
) -> dict:
    from .build import verify_metadata

    dataset_id, version_id = str(record.datasetId), str(record.latestVersionId)
    paths = list_checkpoint_paths(dataset_id, version_id, storage)
    record.checkpointPaths = paths
    if not paths:
        record.cleanupIntent = None
        return {"outcome": "skipped", "message": "No staged source files remain"}
    receipt, manifest = record.buildReceipt, record.inspection
    if (
        record.status != "ready"
        or record.processedVersionId != record.latestVersionId
        or record.zarrUri is None
        or manifest is None
        or manifest.datasetId != record.datasetId
        or manifest.datasetVersionId != record.latestVersionId
        or receipt is None
        or receipt.get("datasetVersionId") != version_id
        or receipt.get("sourceSha256") != manifest.sourceSha256
        or receipt.get("zarrUri") != record.zarrUri
        or not receipt.get("verifiedAt")
        or not receipt.get("verification", {}).get("countsTMatches")
    ):
        return {
            "outcome": "unmetPrerequisite",
            "message": "Cleanup requires a ready Scarf store verified for this version and source checksum",
        }
    prefix = dataset_prefix(record.cytebaseId)
    converted = storage.read_json(f"{prefix}/scarf_ingest.json")
    if (
        converted is None
        or converted.get("status") != "done"
        or converted.get("datasetId") != dataset_id
        or converted.get("datasetVersionId") != version_id
        or converted.get("sourceSha256") != manifest.sourceSha256
        or converted.get("zarrPath") != record.zarrUri
        or converted.get("verification") != receipt["verification"]
    ):
        return {
            "outcome": "unmetPrerequisite",
            "message": "Scarf provenance does not match the committed build receipt",
        }
    completion = get_completion(record, storage)
    if completion is not None and completion["sourceSha256"] != manifest.sourceSha256:
        return {
            "outcome": "unmetPrerequisite",
            "message": "Staged bytes and the committed Scarf store have different source checksums",
        }
    if completion is None:
        intent = record.cleanupIntent
        if (
            intent is None
            or intent.get("datasetId") != dataset_id
            or intent.get("datasetVersionId") != version_id
            or intent.get("sourceSha256") != manifest.sourceSha256
            or not isinstance(intent.get("paths"), list)
            or not set(paths).issubset(intent["paths"])
        ):
            return {
                "outcome": "unmetPrerequisite",
                "message": "Staging has no matching completion receipt or cleanup intent; complete the download or preview manual pruning",
            }
    progress(
        "verifying_cleanup", message="Checking published store metadata and build proof"
    )
    retry(
        lambda: verify_metadata(
            record.zarrUri,
            manifest.model_dump(mode="json"),
            storage_options={"token": storage.token, "skip_instance_cache": True},
            verification=receipt["verification"],
        ),
        progress=progress,
    )
    if completion is not None:
        record.cleanupIntent = {
            "datasetId": dataset_id,
            "datasetVersionId": version_id,
            "sourceSha256": manifest.sourceSha256,
            "paths": paths,
            "startedAt": datetime.now(UTC).isoformat(),
        }
        assert_owner()
        storage.write_json(f"{prefix}/dataset.json", record.model_dump(mode="json"))
    record.checkpointPaths = remove_staging(
        dataset_id,
        version_id,
        storage,
        exact_paths=paths,
        progress=progress,
        assert_owner=assert_owner,
    )
    if record.checkpointPaths:
        raise RuntimeError(
            "Staging cleanup left these paths: " + ", ".join(record.checkpointPaths)
        )
    record.cleanupIntent = None
    return {"outcome": "succeeded"}


def _prune(
    record: DatasetRecord,
    request: dict,
    storage: Bucket,
    progress: Callable,
    assert_owner: Callable[[], None],
) -> dict:
    validate_prune_owner(record, request)
    for item in request["pruneVersions"]:
        current = list_checkpoint_paths(
            str(record.datasetId), item["datasetVersionId"], storage
        )
        if set(current) != set(item["paths"]):
            raise ValueError(
                "Staging inventory changed; preview and review exact paths again"
            )
    deleted = []
    for item in request["pruneVersions"]:
        remaining = remove_staging(
            str(record.datasetId),
            item["datasetVersionId"],
            storage,
            exact_paths=item["paths"],
            progress=progress,
            assert_owner=assert_owner,
        )
        if remaining:
            raise RuntimeError("Prune incomplete; review remaining staging paths")
        deleted.extend(item["paths"])
    return {"outcome": "succeeded", "deletionPaths": deleted}


def run_cleanup(
    record: DatasetRecord,
    request: dict,
    storage: Bucket,
    progress: Callable,
    assert_owner: Callable[[], None],
) -> dict:
    """Cleanup failures retain ready stores and expose remaining staged files."""
    record.checkpointCleanupError = None
    is_prune = bool(request.get("pruneVersions"))
    try:
        assert_owner()
        return (
            _prune(record, request, storage, progress, assert_owner)
            if is_prune
            else _cleanup(record, storage, progress, assert_owner)
        )
    except Exception as error:
        record.checkpointCleanupError = error_message(error)
        raise
    finally:
        try:
            versions = (
                [item["datasetVersionId"] for item in request["pruneVersions"]]
                if is_prune
                else [str(record.latestVersionId)]
            )
            if (
                is_prune
                and record.status == "downloaded"
                and str(record.latestVersionId) in versions
                and get_completion(record, storage) is None
            ):
                record.status = "registered"
            record.checkpointPaths = [
                path
                for version in versions
                for path in list_checkpoint_paths(
                    str(record.datasetId), version, storage
                )
            ]
        except Exception as error:
            logging.warning(
                "Could not refresh staging inventory: %s", error_message(error)
            )
