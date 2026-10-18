"""Tests for tools_filtering.py against agent_dev/sample_fixture.csv."""

import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_filtering import make_tools as make_filtering_tools


def test_variance_filter_tool():
    # 1. Instantiate fresh session
    session = SessionState()

    # 2. Load dataset via premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # 3. Build global DTM first
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # 4. Instantiate filtering tools
    filtering_tools = make_filtering_tools(session)
    variance_filter_tool = next(t for t in filtering_tools if t.name == "variance_filter_tool")

    # 5. Invoke variance_filter_tool with threshold=0.15
    result = variance_filter_tool.invoke({"threshold": 0.15})

    # 6. Assert against Known Answer #5 in TEST_FIXTURE.md
    assert result["n_kept"] == 4
    assert result["n_removed"] == 1

    variances = result["variances"]
    assert variances["alpha"] == pytest.approx(1.1875, abs=1e-3)
    assert variances["gamma"] == pytest.approx(1.109375, abs=1e-3)
    assert variances["delta"] == pytest.approx(0.25, abs=1e-3)
    assert variances["beta"] == pytest.approx(0.234375, abs=1e-3)
    assert variances["always"] == pytest.approx(0.109375, abs=1e-3)

    # Check that full_report was stored in session.results_store
    stored_result = session.results_store[result["result_id"]]
    assert stored_result["full_report"] is not None
    df_report = stored_result["full_report"]
    assert list(df_report.columns) == ["feature", "variance"]
    # Check that full_report is sorted descending by variance
    report_variances = df_report["variance"].tolist()
    assert report_variances == sorted(report_variances, reverse=True)


def test_pearson_filter_tool():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    filtering_tools = make_filtering_tools(session)
    pearson_filter_tool = next(t for t in filtering_tools if t.name == "pearson_filter_tool")

    result = pearson_filter_tool.invoke({"target_class": "catB", "threshold": 0.0})

    # Assert correlations against Known Answer #6 in TEST_FIXTURE.md
    corrs = result["correlations"]
    assert corrs["delta"] == pytest.approx(1.0000, abs=1e-3)
    assert corrs["gamma"] == pytest.approx(0.8307, abs=1e-3)
    assert corrs["always"] == pytest.approx(0.3780, abs=1e-3)
    assert corrs["alpha"] == pytest.approx(-0.6882, abs=1e-3)
    assert corrs["beta"] == pytest.approx(-0.7746, abs=1e-3)

    # Check full_report stored
    stored_result = session.results_store[result["result_id"]]
    assert stored_result["full_report"] is not None
    df_report = stored_result["full_report"]
    assert list(df_report.columns) == ["feature", "pearson_r"]

    # Check that full_report is sorted by absolute pearson_r descending
    abs_corrs = df_report["pearson_r"].abs().tolist()
    assert abs_corrs == sorted(abs_corrs, reverse=True)


def test_spearman_filter_tool():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    filtering_tools = make_filtering_tools(session)
    spearman_filter_tool = next(t for t in filtering_tools if t.name == "spearman_filter_tool")

    result = spearman_filter_tool.invoke({"target_class": "catB", "threshold": 0.0})

    # Verify result structure and expected correlation values from TEST_FIXTURE.md #6
    assert "correlations" in result
    corrs = result["correlations"]

    expected_spearman = {
        "alpha": -0.7500,
        "always": 0.3780,
        "beta": -0.7746,
        "delta": 1.0000,
        "gamma": 0.9363,
    }

    for term, exp_val in expected_spearman.items():
        assert term in corrs, f"Expected term {term} in correlation results"
        assert corrs[term] == pytest.approx(exp_val, abs=1e-3)

    # Check full_report stored
    stored_result = session.results_store[result["result_id"]]
    assert stored_result["full_report"] is not None
    df_report = stored_result["full_report"]
    assert list(df_report.columns) == ["feature", "spearman_r"]

    # Check that full_report is sorted by absolute spearman_r descending
    abs_corrs = df_report["spearman_r"].abs().tolist()
    assert abs_corrs == sorted(abs_corrs, reverse=True)
