"""Tests for tools_data.py against agent_dev/sample_fixture.csv."""

import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools


def test_check_missing_tool():
    # 1. Instantiate the single session object
    session = SessionState()

    # 2. Load premade tools to populate session.dataframe via load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load fixture dataset into session
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # 3. Instantiate student tools from tools_data.py bound to the same session
    tools = make_tools(session)
    check_missing_tool = next(t for t in tools if t.name == "check_missing_tool")

    # 4. Invoke check_missing_tool
    result = check_missing_tool.invoke({})

    # 5. Assert against Known Answer #2 from TEST_FIXTURE.md
    # Known Answer #2: Exactly 1 row has missing/empty text (the last row, index 7).
    assert "missing_count" in result
    assert result["missing_count"] == 1
    assert result["missing_indices"] == [7]


def test_check_duplicates_tool_without_drop():
    # Fresh session to prevent state leakage across test functions
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    check_duplicates_tool = next(t for t in tools if t.name == "check_duplicates_tool")

    # Check duplicates without dropping
    result = check_duplicates_tool.invoke({"drop": False})

    # Assert Known Answer #3 (check phase):
    # Exactly 1 duplicate (row 6 is an exact copy of row 5)
    assert result["duplicate_count"] == 1
    assert result["duplicate_indices"] == [6]
    assert result["dropped"] is False
    assert result["remaining_documents"] == 8


def test_check_duplicates_tool_with_drop():
    # Fresh session for isolated test
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    check_duplicates_tool = next(t for t in tools if t.name == "check_duplicates_tool")

    # Check duplicates with drop=True
    result = check_duplicates_tool.invoke({"drop": True})

    # Assert Known Answer #3 (drop phase):
    # Dropping with keep=False leaves exactly 6 rows (rows 5 and 6 are both gone)
    assert result["dropped"] is True
    assert result["remaining_documents"] == 6
    assert len(session.dataframe) == 6


def test_inspect_data_tool():
    # Fresh session for isolated test
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    inspect_data_tool = next(t for t in tools if t.name == "inspect_data_tool")

    # Test with n=3
    result = inspect_data_tool.invoke({"n": 3})

    # Assert requirements:
    # 1. returns 3 records when n=3
    assert "head_records" in result
    assert len(result["head_records"]) == 3

    # 2. total_rows is 8 on sample_fixture.csv
    assert "total_rows" in result
    assert result["total_rows"] == 8


def test_describe_data_tool():
    # Fresh session for isolated test
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    describe_data_tool = next(t for t in tools if t.name == "describe_data_tool")

    result = describe_data_tool.invoke({})

    # Assert against Known Answer #18 in TEST_FIXTURE.md
    overall = result["overall_stats"]
    assert overall["count"] == 8
    assert overall["mean"] == pytest.approx(19.875, abs=1e-3)
    assert overall["std"] == pytest.approx(9.4330, abs=1e-3)
    assert overall["min"] == 0
    assert overall["25%"] == pytest.approx(17.75, abs=1e-3)
    assert overall["50%"] == pytest.approx(20.5, abs=1e-3)
    assert overall["75%"] == pytest.approx(25.25, abs=1e-3)
    assert overall["max"] == 30

    per_cat = result["per_category_stats"]
    assert "catA" in per_cat
    assert per_cat["catA"]["count"] == 4
    assert per_cat["catA"]["mean"] == pytest.approx(17.25, abs=1e-3)

    assert "catB" in per_cat
    assert per_cat["catB"]["count"] == 4
    assert per_cat["catB"]["mean"] == pytest.approx(22.5, abs=1e-3)

    # Verify pending_figure is set
    assert session.pending_figure is not None


def test_tokenize_tool():
    # Fresh session for isolated test
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })

    tools = make_tools(session)
    tokenize_tool = next(t for t in tools if t.name == "tokenize_tool")

    result = tokenize_tool.invoke({})

    # Assert Known Answer #10 in TEST_FIXTURE.md
    assert result["total_documents"] == 8
    assert result["total_tokens"] == 27

    expected_unigrams = [
        ["always", "alpha", "alpha", "alpha", "beta"],
        ["always", "alpha", "alpha", "beta"],
        ["always", "alpha", "beta"],
        ["always", "gamma", "gamma", "gamma", "delta"],
        ["always", "gamma", "gamma", "delta"],
        ["always", "gamma", "delta"],
        ["always", "gamma", "delta"],
        [],
    ]

    assert "unigrams" in session.dataframe.columns
    actual_unigrams = session.dataframe["unigrams"].tolist()
    assert actual_unigrams == expected_unigrams
    assert session.artifacts["unigrams"] == expected_unigrams


def test_sample_data_tool():
    # Fresh session for isolated test
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })

    tools = make_tools(session)
    sample_data_tool = next(t for t in tools if t.name == "sample_data_tool")

    # Assert Known Answer #13 in TEST_FIXTURE.md
    result = sample_data_tool.invoke({"n": 4, "random_state": 42})

    assert result["n_sampled"] == 4
    assert result["sampled_indices"] == [1, 5, 0, 7]
    assert len(result["sampled_head"]) <= 5

    # Check that session.dataframe was NOT mutated (still has all 8 rows)
    assert len(session.dataframe) == 8

    # Check session.artifacts['sampled_dataframe']
    assert "sampled_dataframe" in session.artifacts
    sampled_df = session.artifacts["sampled_dataframe"]
    assert sampled_df.shape[0] == 4
    assert sampled_df.index.tolist() == [1, 5, 0, 7]


def test_list_files_tool():
    session = SessionState()
    tools = make_tools(session)
    list_files_tool = next(t for t in tools if t.name == "list_files_tool")

    # Assert Known Answer #11 in TEST_FIXTURE.md
    res = list_files_tool.invoke({"subdirectory": "newdataset"})

    assert res["subdirectory"] == "newdataset"
    assert res["directories"] == []
    assert res["files"] == ["Reddit-stock-sentiment.csv"]
    assert res["n_directories"] == 0
    assert res["n_files"] == 1

    # Assert path containment check rejects directories outside repo root
    res_outside = list_files_tool.invoke({"subdirectory": "../.."})
    assert "error" in res_outside
    assert "outside the repository root" in res_outside["error"]
