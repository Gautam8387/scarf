"""Supported local HTML report contracts for completed agent workflows."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import pytest
import zarr

import scarf.agent as agent_api
import scarf.agent.report as report_module
import scarf.agent.orchestrator.main as orchestrator_main
from scarf.agent import (
    AgentWorkflowRun,
    AutomatedWorkflowConfig,
    AutomatedWorkflowRequest,
    AutomatedWorkflowResult,
    FinalAnalysisHandoff,
    create_agent_workflow,
    generate_agent_report,
    list_agent_workflows,
    load_agent_workflow,
)


def _workflow(
    *,
    workspace: str | None = None,
    status: Literal["completed", "running"] = "completed",
) -> AgentWorkflowRun:
    return AgentWorkflowRun(
        workflowRunId="report-workflow",
        workspace=workspace,
        createdAtNs=1,
        finalizedAtNs=2 if status != "running" else 0,
        status=status,
        finalizationMessage="analysis completed" if status != "running" else "",
        analysisStore="data.zarr",
        datasetFingerprints={"RNA": "dataset-rna"},
    )


def _reports(study_context: str) -> dict[str, list[dict[str, Any]]]:
    run_info = {
        "agentName": "data_enrichment",
        "runId": "provider-run",
        "modelName": "test-model",
        "durationSeconds": 1.5,
        "usage": {
            "requests": 2,
            "toolCalls": 1,
            "inputTokens": 20,
            "outputTokens": 5,
            "totalTokens": 25,
        },
    }
    candidate = {
        "candidateId": "refined",
        "phase": "refined",
        "status": "done",
        "eligible": True,
        "parameters": {
            "reductionMethod": "pca",
            "dimensions": 21,
            "neighborsK": 11,
            "leidenResolution": 0.75,
            "useHarmony": False,
        },
        "metrics": {
            "nClusters": 7,
            "minClusterCells": 42,
            "graphSilhouetteMedian": 0.343,
        },
    }
    return {
        "data_enrichment": [
            {
                "status": "done",
                "studyContextSummary": {
                    "studyContext": study_context,
                    "organismReferences": ["human"],
                    "tissueReferences": ["blood"],
                },
                "policies": [{"assay": "RNA", "policyId": "rna-default"}],
                "runInfo": run_info,
            }
        ],
        "experimental_context": [
            {
                "status": "done",
                "decision": {"batchCorrection": {"action": "skip"}},
                "cellQc": {"action": "globalGaussian", "driverAssay": "RNA"},
            }
        ],
        "parameter_tuning": [
            {
                "status": "done",
                "fromAssay": "RNA",
                "totalCandidates": 2,
                "recommendedByAssay": {"RNA": "refined"},
                "rationale": "The refined candidate balanced cluster viability.",
                "stopReason": "The bounded refinement completed.",
                "assayReports": {
                    "RNA": {
                        "recommendedCandidateId": "refined",
                        "confidence": "medium",
                        "evaluations": [candidate],
                        "comparisons": [
                            {
                                "candidateId": "baseline",
                                "summary": (
                                    "The refined candidate retained larger "
                                    "minimum clusters."
                                ),
                                "evidenceIds": ["candidate:refined:clusters"],
                            }
                        ],
                        "searchPlan": {
                            "status": "refine",
                            "objectives": ["Test an intermediate resolution."],
                        },
                    }
                },
            }
        ],
        "biological_interpretation": [
            {
                "status": "done",
                "clusterInterpretations": [
                    {
                        "clusterId": "0",
                        "proposedIdentity": "T cell",
                        "identityIsHypothesis": True,
                    }
                ],
                "treatmentObservations": [],
                "followUps": ["Validate the proposed identities."],
            }
        ],
    }


def _patch_completed_workflow(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    *,
    workspace: str | None = None,
    study_context: str = "A human blood study.",
    plots: bool = True,
) -> Path:
    group = zarr.open_group(str(root), mode="w", zarr_format=3)
    if workspace is not None:
        group.create_group(workspace)
    workflow = _workflow(workspace=workspace)
    final = FinalAnalysisHandoff.get_example().model_copy(
        update={"workflowRunId": workflow.workflowRunId}
    )
    result = AutomatedWorkflowResult(
        status="completed",
        currentStage="biological_interpretation",
        zarrPath=str(root),
        workflowRun=workflow,
        finalAnalysis=final,
    )
    request = AutomatedWorkflowRequest(
        sourcePath="input.h5ad",
        zarrPath=str(root),
        studyContext=study_context,
        workspace=workspace,
    )
    request_record = SimpleNamespace(
        request=request,
        config=AutomatedWorkflowConfig(),
    )

    monkeypatch.setattr(
        report_module,
        "load_agent_workflow",
        lambda *_a, **_k: workflow,
    )
    monkeypatch.setattr(report_module, "_open_datastore", lambda *_a, **_k: object())
    monkeypatch.setattr(
        report_module,
        "_load_completed_result",
        lambda *_a, **_k: ("agents/orchestrations", result, request_record),
    )
    monkeypatch.setattr(
        report_module,
        "_collect_reports",
        lambda *_a, **_k: _reports(study_context),
    )
    monkeypatch.setattr(
        report_module,
        "_collect_history",
        lambda *_a, **_k: (
            [
                {
                    "stage": "parameter_tuning",
                    "status": "done",
                    "durationSeconds": 3.5,
                    "actions": ["evaluate_refined_candidate"],
                    "reportCount": 1,
                    "artifactCount": 4,
                    "artifacts": {
                        "selectedGraph": {
                            "scope": "assay",
                            "assay": "RNA",
                            "kind": "connectivity_map",
                            "artifactId": "a" * 64,
                        }
                    },
                    "parentAttempts": ["preprocessing:attempt-1"],
                    "questionIds": [],
                    "noteCount": 0,
                    "errorType": None,
                }
            ],
            [],
        ),
    )

    def collect_artifacts(
        _store: object,
        _result: AutomatedWorkflowResult,
        plot_dir: Path,
    ) -> tuple[dict[str, int], list[dict[str, Any]], dict[str, str], list[str]]:
        if not plots:
            return (
                {"0": 3, "1": 2},
                [],
                {},
                ["umapClusters: ImportError: plotting dependencies are unavailable"],
            )
        plot_dir.mkdir(parents=True, exist_ok=True)
        (plot_dir / "final_umap.png").write_bytes(b"png")
        (plot_dir / "final_umap.png.json").write_text(
            '{"artifact":"umap"}\n', encoding="utf-8"
        )
        return (
            {"0": 3, "1": 2},
            [{"group_id": "0", "feature_name": "CD3D", "score": 8.5}],
            {"umapClusters": "plots/final_umap.png"},
            [],
        )

    monkeypatch.setattr(report_module, "_collect_final_artifacts", collect_artifacts)
    return root


def test_public_report_generates_branded_readable_html_and_relative_plots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _patch_completed_workflow(
        monkeypatch,
        tmp_path / "data.zarr",
        study_context='Human blood <script>alert("unsafe")</script> & treatment.',
    )
    immutable_record = root / "agents/runs/report-workflow/workflow.json"
    immutable_record.parent.mkdir(parents=True)
    immutable_record.write_bytes(b'{"immutable":true}\n')

    report_path = generate_agent_report(root, "report-workflow")
    markup = report_path.read_text(encoding="utf-8")

    assert agent_api.generate_agent_report is generate_agent_report
    assert report_path == root / "agents/runs/report-workflow/report/index.html"
    assert immutable_record.read_bytes() == b'{"immutable":true}\n'
    assert 'href="https://www.nygen.io/"' in markup
    assert ">Nygen Analytics</a>" in markup
    assert 'href="https://www.nygen.io/products/scarfweb"' in markup
    assert (
        "Distributed, secure infrastructure for intuitive secondary analysis, "
        "browser-native."
    ) in markup
    assert "Human blood &lt;script&gt;alert" in markup
    assert '<script>alert("unsafe")</script>' not in markup
    assert "Parameter tuning and graph selection" in markup
    assert "refined" in markup
    assert "0.343" in markup
    assert "The refined candidate retained larger minimum clusters." in markup
    assert "Stage artifact inventory" in markup
    assert "connectivity_map" in markup
    assert "Recorded totals" in markup
    assert "evaluate_refined_candidate" in markup
    assert 'src="plots/final_umap.png"' in markup
    assert 'href="plots/final_umap.png.json"' in markup
    assert (report_path.parent / "plots/final_umap.png").read_bytes() == b"png"


def test_report_uses_workspace_path_and_can_be_regenerated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _patch_completed_workflow(
        monkeypatch,
        tmp_path / "data.zarr",
        workspace="analysis",
        study_context="First context",
    )

    first = generate_agent_report(root, "report-workflow", workspace="analysis")
    assert first == (root / "analysis/agents/runs/report-workflow/report/index.html")
    first_markup = first.read_text(encoding="utf-8")
    assert "First context" in first_markup

    monkeypatch.setattr(
        report_module,
        "_collect_reports",
        lambda *_a, **_k: _reports("Regenerated context"),
    )
    second = generate_agent_report(root, "report-workflow", workspace="analysis")

    assert second == first
    second_markup = second.read_text(encoding="utf-8")
    assert "Regenerated context" in second_markup
    assert second_markup != first_markup


def test_report_remains_available_when_optional_plots_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _patch_completed_workflow(
        monkeypatch,
        tmp_path / "data.zarr",
        plots=False,
    )

    report_path = generate_agent_report(root, "report-workflow")
    markup = report_path.read_text(encoding="utf-8")

    assert report_path.is_file()
    assert "No plots could be rendered" in markup
    assert "plotting dependencies are unavailable" in markup
    assert "Final cluster sizes" in markup


def test_report_rejects_remote_and_non_completed_workflows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="local filesystem"):
        generate_agent_report("s3://bucket/data.zarr", "report-workflow")

    root = tmp_path / "data.zarr"
    zarr.open_group(str(root), mode="w", zarr_format=3)
    running = _workflow(status="running")
    monkeypatch.setattr(
        report_module,
        "load_agent_workflow",
        lambda *_a, **_k: running,
    )
    monkeypatch.setattr(report_module, "_open_datastore", lambda *_a, **_k: object())

    with pytest.raises(RuntimeError, match="completed workflows"):
        generate_agent_report(root, running.workflowRunId)


def test_orchestrator_generates_only_completed_local_reports_non_fatally(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    generated: list[tuple[object, str]] = []
    local_store = SimpleNamespace(z=object())
    completed = _workflow()

    monkeypatch.setattr(
        orchestrator_main,
        "zarr_root_path",
        lambda _store: tmp_path / "data.zarr",
    )
    monkeypatch.setattr(
        report_module,
        "generate_agent_report",
        lambda target, workflow_run_id: (
            generated.append((target, workflow_run_id))
            or tmp_path / "data.zarr/agents/runs/report-workflow/report/index.html"
        ),
    )

    orchestrator_main._generate_completed_report(local_store, completed)

    assert generated == [(local_store, completed.workflowRunId)]
    assert "Agent workflow report:" in capsys.readouterr().out

    orchestrator_main._generate_completed_report(
        local_store,
        _workflow(status="running"),
    )
    monkeypatch.setattr(orchestrator_main, "zarr_root_path", lambda _store: None)
    orchestrator_main._generate_completed_report(local_store, completed)
    assert len(generated) == 1

    monkeypatch.setattr(
        orchestrator_main,
        "zarr_root_path",
        lambda _store: tmp_path / "data.zarr",
    )
    monkeypatch.setattr(
        report_module,
        "generate_agent_report",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("plot failed")),
    )
    orchestrator_main._generate_completed_report(local_store, completed)


def test_derived_report_files_do_not_change_workflow_record_discovery(
    tmp_path: Path,
) -> None:
    root = zarr.open_group(str(tmp_path / "data.zarr"), mode="w", zarr_format=3)
    root.create_group("cellData")
    assay = root.create_group("RNA")
    assay.attrs["is_assay"] = True
    assay.attrs["dataset_fingerprint"] = "dataset-rna"
    workflow = create_agent_workflow(root, workflow_run_id="report-workflow")
    report_dir = (
        tmp_path / "data.zarr" / "agents" / "runs" / workflow.workflowRunId / "report"
    )
    plot_dir = report_dir / "plots"
    plot_dir.mkdir(parents=True)
    (report_dir / "index.html").write_text("<html></html>", encoding="utf-8")
    (plot_dir / "final_umap.png").write_bytes(b"png")
    (plot_dir / "final_umap.png.json").write_text("{}\n", encoding="utf-8")

    assert load_agent_workflow(root, workflow.workflowRunId) == workflow
    assert list_agent_workflows(root, include_incomplete=True) == [workflow]
