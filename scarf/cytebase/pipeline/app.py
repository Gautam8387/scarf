"""Four Modal stage pools, a queued orchestrator, and the development HTTP API."""

import asyncio
import logging
import os
import subprocess
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from threading import Event, Lock, Thread
from time import monotonic

import modal
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from .._storage import Bucket, dataset_prefix, error_message
from .catalog import load_record, select_dataset_ids
from .models import ProcessRequest, PruneRequest, RegisterRequest

app = modal.App("cellxgene-cytebase")
secret = modal.Secret.from_name(
    "scarf-env", required_keys=["HF_TOKEN", "CYTEBASE_BUCKET"]
)
progress_store = modal.Dict.from_name("cytebase-progress", create_if_missing=True)
RUN_PATH = "_internal/pipeline.json"


def _limit(name: str) -> int:
    value = int(os.environ.get(name, "4"))
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


LIMITS = {
    name: _limit(f"CYTEBASE_{name}_CONTAINERS")
    for name in ("DOWNLOAD", "BUILD", "CLEANUP")
}
_SHA = (
    subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True
    ).strip()
    if modal.is_local()
    else "remote"
)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("build-essential", "git", "libfftw3-dev", "libmetis-dev", "libtbb-dev")
    .uv_sync(groups=["cytebase"], frozen=True, extra_options="--no-default-groups")
    .add_local_python_source(
        "scarf",
        copy=True,
        ignore=lambda path: "datasets" in path.parts or path.suffix != ".py",
    )
    .env(
        {
            "CYTEBASE_PIPELINE_VERSION": _SHA,
            **{
                f"CYTEBASE_{key}_CONTAINERS": str(value)
                for key, value in LIMITS.items()
            },
        }
    )
)


def _storage() -> Bucket:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("Missing required environment variable: HF_TOKEN")
    return Bucket(token=token)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _owner(storage: Bucket, run_id: str, key: str, call_id: str) -> None:
    state = storage.read_json(RUN_PATH) or {}
    child = state.get("children", {}).get(key, {})
    if (
        state.get("runId") != run_id
        or state.get("state") != "running"
        or child.get("state") not in {"pending", "running"}
        or child.get("callId") not in {None, call_id}
    ):
        raise RuntimeError("This worker no longer owns the current pipeline stage")
    if not child:
        raise RuntimeError("Pipeline has no reservation for this worker")


@contextmanager
def _progress(record, stage: str):
    """Progress is display-only; expiry never changes durable dataset readiness."""
    key = f"{record.runId}:{record.cytebaseId}:{stage}"
    lock, stopped = Lock(), Event()
    value = {
        "runId": record.runId,
        "callId": record.callId,
        "stage": stage,
        "attempt": record.attempt,
        "heartbeatAt": _now(),
        "progress": None,
    }

    def publish():
        with lock:
            value["heartbeatAt"] = _now()
            snapshot = dict(value)
        try:
            progress_store.put(key, snapshot)
        except Exception as error:
            logging.warning("Progress update failed: %s", error_message(error))

    def update(name: str, **counters):
        completed, total = counters.get("completed"), counters.get("total")
        counters["percent"] = (
            min(100, 100 * completed / total)
            if completed is not None and total
            else None
        )
        with lock:
            changed = name != value["stage"]
            value.update(stage=name, progress=counters)
        if changed:
            publish()

    def heartbeat():
        while not stopped.wait(15):
            publish()

    publish()
    thread = Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        yield update
    finally:
        stopped.set()
        thread.join()


def _execute(stage: str, cytebase_id: str, run_id: str, request: dict) -> dict:
    from .build import run_build
    from .cleanup import run_cleanup
    from .download import run_download

    storage = _storage()
    call_id = modal.current_function_call_id()
    key = f"{cytebase_id}:{stage}"

    def check():
        _owner(storage, run_id, key, call_id)

    check()
    record = load_record(storage, cytebase_id)
    state = storage.read_json(RUN_PATH)
    if str(record.latestVersionId) != state["children"][key]["datasetVersionId"]:
        raise ValueError("Registered version changed after submission")
    record.attempt += int(record.runId != run_id)
    record.runId, record.callId, record.stage = run_id, call_id, stage
    record.pipelineVersion = os.environ["CYTEBASE_PIPELINE_VERSION"]
    record.stageOutcome, record.error = "running", None
    record.startedAt = datetime.now(UTC)
    record.checkpointCleanupError = None
    if stage not in {"cleanup", "prune"}:
        record.needsInput = None
    storage.write_json(
        f"{dataset_prefix(cytebase_id)}/dataset.json", record.model_dump(mode="json")
    )
    started = monotonic()
    try:
        with _progress(record, stage) as progress:
            storage.progress = progress
            result = {
                "download": run_download,
                "build": run_build,
                "cleanup": run_cleanup,
                "prune": run_cleanup,
            }[stage](record, request | {"stage": stage}, storage, progress, check)
    except Exception as error:
        message = error_message(error)
        result = {"outcome": "failed", "message": message}
        if stage in {"cleanup", "prune"}:
            record.checkpointCleanupError = message
        elif not (stage == "download" and record.status == "ready"):
            record.status = "failed"
    record.stageOutcome = result["outcome"]
    if result["outcome"] not in {"succeeded", "skipped"}:
        record.error = result.get("message")
    record.timings[f"{stage}Seconds"] = monotonic() - started
    record.updatedAt = datetime.now(UTC)
    check()
    storage.write_json(
        f"{dataset_prefix(cytebase_id)}/dataset.json", record.model_dump(mode="json")
    )
    return result | {
        "cytebaseId": cytebase_id,
        "status": record.status,
        "record": record.model_dump(mode="json"),
    }


@app.function(
    image=image,
    secrets=[secret],
    cpu=1,
    memory=4096,
    timeout=86400,
    retries=0,
    max_containers=1,
)
def build_catalog(request: dict, run_id: str) -> dict:
    from .catalog import run_catalog

    storage = _storage()

    def check():
        _owner(storage, run_id, "catalog", modal.current_function_call_id())

    check()
    return run_catalog(request, storage, check)


@app.function(
    image=image,
    secrets=[secret],
    cpu=1,
    memory=4096,
    timeout=86400,
    retries=0,
    max_containers=LIMITS["DOWNLOAD"],
)
def download_h5ad(cytebase_id: str, run_id: str, request: dict) -> dict:
    return _execute("download", cytebase_id, run_id, request)


@app.function(
    image=image,
    secrets=[secret],
    cpu=(2, 8),  # request 2 cpus with soft limit of 8
    memory=(8192, 32768),  # request 8 GB with soft limit of 32 GB
    timeout=86400,
    retries=0,
    max_containers=LIMITS["BUILD"],
)
def build_scarf(cytebase_id: str, run_id: str, request: dict) -> dict:
    return _execute("build", cytebase_id, run_id, request)


@app.function(
    image=image,
    secrets=[secret],
    cpu=2,
    memory=4096,
    timeout=14400,
    retries=0,
    max_containers=LIMITS["CLEANUP"],
)
def cleanup(cytebase_id: str, run_id: str, request: dict) -> dict:
    return _execute(request.get("stage", "cleanup"), cytebase_id, run_id, request)


def _reset(storage: Bucket, request: dict) -> dict:
    state = storage.read_json(RUN_PATH) or {}
    if not request.get("workersDrained") or state.get("runId") != request.get(
        "expectedRunId"
    ):
        raise ValueError(
            "Reset requires the exact expected run ID and explicit workersDrained confirmation"
        )
    ids = [
        state.get("callId"),
        *(child.get("callId") for child in state.get("children", {}).values()),
    ]
    for call_id in filter(None, ids):
        try:
            modal.FunctionCall.from_id(call_id).get(timeout=0)
        except modal.exception.FunctionTimeoutError:
            pass
        except TimeoutError as error:
            raise ValueError(
                f"Call {call_id} is still running; drain workers before reset"
            ) from error
        except Exception as error:
            # Modal re-raises original worker exceptions, not just RemoteError.
            # Failed or unavailable results rely on the required operator drain
            # confirmation; no elapsed-time inference authorizes this reset.
            logging.warning(
                "Using explicit worker-drain confirmation for call %s: %s",
                call_id,
                error_message(error),
            )
    affected = {
        child["cytebaseId"]
        for child in state.get("children", {}).values()
        if child.get("cytebaseId")
    }
    for key in affected:
        record = load_record(storage, key)
        if record.runId == state["runId"] and record.stageOutcome == "running":
            record.stageOutcome = "failed"
            record.error = (
                "Interrupted run reset after explicit worker-drain confirmation"
            )
            if record.status == "processing":
                record.status = "failed"
            record.updatedAt = datetime.now(UTC)
            storage.write_json(
                f"{dataset_prefix(key)}/dataset.json", record.model_dump(mode="json")
            )
    state.update(state="reset", updatedAt=_now())
    storage.write_json(RUN_PATH, state)
    return {"runId": state["runId"], "state": "reset"}


@app.function(
    image=image,
    secrets=[secret],
    cpu=1,
    memory=2048,
    timeout=86400,
    retries=0,
    max_containers=1,
)
async def run_pipeline(action: str, request: dict) -> dict:
    """Queue submissions, but advance datasets independently inside each submission."""
    from .catalog import list_records
    from .download import prune_groups

    storage = _storage()
    if action == "reset":
        return await asyncio.to_thread(_reset, storage, request)
    previous = await asyncio.to_thread(storage.read_json, RUN_PATH) or {}
    if previous and previous.get("state") not in {"completed", "failed", "reset"}:
        raise RuntimeError(
            f"Run {previous.get('runId')} has unresolved workers; drain and explicitly reset it before resubmitting"
        )
    groups = {}
    if action == "prune":
        records = await asyncio.to_thread(list_records, storage)
        groups = await asyncio.to_thread(
            prune_groups, request["approvedPaths"], storage, records, previous
        )
    elif action not in {"register", "catalog"}:
        job = ProcessRequest.model_validate(request)
        keys = await asyncio.to_thread(select_dataset_ids, job, storage)
        groups = {key: request for key in keys}
    state = {
        "runId": modal.current_function_call_id(),
        "callId": modal.current_function_call_id(),
        "action": action,
        "state": "running",
        "startedAt": _now(),
        "updatedAt": _now(),
        "children": {},
    }
    await asyncio.to_thread(storage.write_json, RUN_PATH, state)
    lock = asyncio.Lock()
    uncertain = False

    async def invoke(function, key, args, reservation):
        nonlocal uncertain
        try:
            async with lock:
                state["children"][key] = reservation | {
                    "callId": None,
                    "state": "pending",
                }
                state["updatedAt"] = _now()
                await asyncio.to_thread(storage.write_json, RUN_PATH, state)
                call = await function.spawn.aio(*args)
                state["children"][key].update(callId=call.object_id, state="running")
                await asyncio.to_thread(storage.write_json, RUN_PATH, state)
            result = await call.get.aio()
            async with lock:
                state["children"][key]["state"] = result.get("outcome", "succeeded")
                state["updatedAt"] = _now()
                await asyncio.to_thread(storage.write_json, RUN_PATH, state)
            return result
        except Exception:
            uncertain = True
            raise

    async def publish(payload):
        return await invoke(
            build_catalog, "catalog", (payload, state["runId"]), {"stage": "catalog"}
        )

    results, dirty = [], {}
    catalog_result, catalog_error = None, None
    try:
        if action in {"register", "catalog"}:
            catalog_result = await publish(request if action == "register" else {})
        else:
            # Refresh before processing without scanning every remote dataset on
            # subsequent snapshots. Only completed stage records are merged.
            catalog_result = await publish({"updates": []})

            async def one(key, payload):
                stages = (
                    ["download", "build", "cleanup"]
                    if action == "process"
                    else [action]
                )
                outcomes = {}
                try:
                    record = await asyncio.to_thread(load_record, storage, key)
                    for stage in stages:
                        arguments = payload | {
                            "operation": action,
                            "stage": stage,
                            "approvedDeletionPaths": [
                                path
                                for path in payload.get("approvedDeletionPaths", [])
                                if path.startswith(f"{dataset_prefix(key)}/")
                            ],
                        }
                        worker = {
                            "download": download_h5ad,
                            "build": build_scarf,
                            "cleanup": cleanup,
                            "prune": cleanup,
                        }[stage]
                        result = await invoke(
                            worker,
                            f"{key}:{stage}",
                            (key, state["runId"], arguments),
                            {
                                "cytebaseId": key,
                                "datasetVersionId": str(record.latestVersionId),
                                "stage": stage,
                            },
                        )
                        dirty[key] = result.pop("record")
                        outcomes[stage] = result
                        if result["outcome"] not in {"succeeded", "skipped"}:
                            break
                    return {
                        "cytebaseId": key,
                        "status": result["status"],
                        "outcome": result["outcome"],
                        "stages": outcomes,
                    }
                except Exception as error:
                    return {
                        "cytebaseId": key,
                        "outcome": "failed",
                        "message": error_message(error),
                        "stages": outcomes,
                    }

            pending = {
                asyncio.create_task(one(key, payload))
                for key, payload in groups.items()
            }
            last_publish = monotonic()
            while pending:
                done, pending = await asyncio.wait(
                    pending,
                    timeout=max(0.1, 60 - (monotonic() - last_publish)),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                results.extend(task.result() for task in done)
                if monotonic() - last_publish >= 60:
                    if dirty and not uncertain:
                        updates, dirty = list(dirty.values()), {}
                        try:
                            catalog_result = await publish({"updates": updates})
                        except Exception as error:
                            catalog_error = error_message(error)
                    last_publish = monotonic()
            if dirty and not uncertain:
                catalog_result = await publish({"updates": list(dirty.values())})
    except Exception as error:
        catalog_error = error_message(error)
    unfinished = any(
        child.get("state") in {"pending", "running", "unknown"}
        for child in state["children"].values()
    )
    state.update(
        state="blocked"
        if uncertain or unfinished
        else "failed"
        if catalog_error
        else "completed",
        updatedAt=_now(),
    )
    await asyncio.to_thread(storage.write_json, RUN_PATH, state)
    result = {
        "callId": state["runId"],
        "state": state["state"],
        "datasets": results,
        "catalog": catalog_result,
        "error": catalog_error,
        "successes": [
            row["cytebaseId"]
            for row in results
            if row["outcome"] in {"succeeded", "skipped"}
        ],
        "failures": [
            row["cytebaseId"]
            for row in results
            if row["outcome"] not in {"succeeded", "skipped"}
        ],
    }
    if action == "register" and catalog_result:
        result["datasets"] = catalog_result.get("registeredDatasets", [])
    return result


def create_web_app() -> FastAPI:
    web = FastAPI(title="Cytebase pipeline (development)")

    def submit(action, payload):
        call = run_pipeline.spawn(action, payload)
        return {"callId": call.object_id}

    @web.post("/collections/register", status_code=202)
    def register(job: RegisterRequest):
        return submit("register", job.model_dump(mode="json"))

    @web.get("/collections")
    def collections():
        from .catalog import list_collection_ids

        return {"collectionIds": list_collection_ids()}

    def submit_datasets(action: str, job: ProcessRequest):
        try:
            keys = select_dataset_ids(job, _storage())
        except ValueError as error:
            raise HTTPException(422, detail=error_message(error)) from error
        return submit(action, job.model_dump(mode="json")) | {
            "datasets": [
                {"cytebaseId": key, "statusUrl": f"/datasets/{key}"} for key in keys
            ]
        }

    @web.post("/datasets/download", status_code=202)
    def download(job: ProcessRequest):
        return submit_datasets("download", job)

    @web.post("/datasets/build", status_code=202)
    def build(job: ProcessRequest):
        return submit_datasets("build", job)

    @web.post("/datasets/cleanup", status_code=202)
    def clean(job: ProcessRequest):
        return submit_datasets("cleanup", job)

    @web.post("/datasets/process", status_code=202)
    def process(job: ProcessRequest):
        return submit_datasets("process", job)

    @web.post("/catalog/build", status_code=202)
    def catalog():
        return submit("catalog", {})

    @web.get("/jobs/{call_id}", response_model=None)
    def job(call_id: str):
        try:
            return modal.FunctionCall.from_id(call_id).get(timeout=0)
        except (
            modal.exception.OutputExpiredError,
            modal.exception.NotFoundError,
            modal.exception.InvalidError,
        ) as error:
            raise HTTPException(
                404, detail="Job result is missing or expired"
            ) from error
        except modal.exception.FunctionTimeoutError as error:
            raise HTTPException(500, detail=error_message(error)) from error
        except TimeoutError:
            return JSONResponse(status_code=202, content={"status": "pending"})
        except Exception as error:
            raise HTTPException(500, detail=error_message(error)) from error

    @web.get("/datasets/{cytebase_id}")
    def dataset(cytebase_id: str, includeReplacementPaths: bool = False):
        storage = _storage()
        try:
            record = load_record(storage, cytebase_id)
        except FileNotFoundError as error:
            raise HTTPException(404, detail=error_message(error)) from error
        except ValueError as error:
            raise HTTPException(422, detail=error_message(error)) from error
        status = {
            key: getattr(record, key)
            for key in (
                "status",
                "stage",
                "stageOutcome",
                "runId",
                "callId",
                "attempt",
                "updatedAt",
                "error",
                "needsInput",
                "checkpointCleanupError",
                "checkpointPaths",
            )
        }
        if record.stageOutcome == "running":
            status.update(
                progress_store.get(f"{record.runId}:{record.cytebaseId}:{record.stage}")
                or {}
            )
        result = {"dataset": record.model_dump(mode="json"), "status": status}
        if includeReplacementPaths:
            from .build import replacement_paths

            result["replacementPaths"] = replacement_paths(record, storage)
        return result

    @web.get("/downloads")
    def downloads():
        from .catalog import list_records
        from .download import downloads as inventory

        storage = _storage()
        return {
            "downloads": inventory(
                storage, list_records(storage), storage.read_json(RUN_PATH)
            )
        }

    @web.post("/downloads/prune", response_model=None)
    def prune(job: PruneRequest):
        if job.preview:
            return {"preview": True, **downloads()}
        from .catalog import list_records
        from .download import prune_groups

        storage = _storage()
        try:
            groups = prune_groups(
                job.approvedPaths,
                storage,
                list_records(storage),
                storage.read_json(RUN_PATH),
            )
        except ValueError as error:
            raise HTTPException(409, detail=error_message(error)) from error
        return JSONResponse(
            status_code=202,
            content=submit("prune", job.model_dump(mode="json"))
            | {
                "datasets": [
                    {"cytebaseId": key, "statusUrl": f"/datasets/{key}"}
                    for key in groups
                ]
            },
        )

    @web.get("/health")
    def health():
        return {"ok": True}

    return web


@app.function(image=image, secrets=[secret], max_containers=1)
@modal.asgi_app()
def web_app() -> FastAPI:
    return create_web_app()
