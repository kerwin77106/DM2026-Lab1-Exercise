"""Tests for tools_dtm.py against agent_dev/sample_fixture.csv."""

import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools


def test_build_dtm_tool():
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

    # 3. Instantiate student tools from tools_dtm.py
    tools = make_tools(session)
    build_dtm_tool = next(t for t in tools if t.name == "build_dtm_tool")

    # 4. Invoke build_dtm_tool
    result = build_dtm_tool.invoke({})

    # 5. Assert against Known Answer #4 in TEST_FIXTURE.md
    assert "vocabulary_size" in result
    assert result["vocabulary_size"] == 5
    assert result["feature_names"] == ["alpha", "always", "beta", "delta", "gamma"]

    assert result["non_zero"] == 21
    assert result["total_elements"] == 40
    assert result["sparsity_pct"] == pytest.approx(47.5, abs=1e-3)

    # Check session state updates
    assert session.feature_matrix is not None
    assert session.feature_matrix.shape == (8, 5)
    assert session.feature_names == ["alpha", "always", "beta", "delta", "gamma"]
    assert "count_vectorizer" in session.artifacts


def test_term_frequency_tool():
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
    build_dtm_tool = next(t for t in tools if t.name == "build_dtm_tool")
    term_frequency_tool = next(t for t in tools if t.name == "term_frequency_tool")

    # Run build_dtm_tool first to populate session.feature_matrix
    build_dtm_tool.invoke({})

    # Invoke term_frequency_tool
    result = term_frequency_tool.invoke({})

    # Assert against Known Answer #14 in TEST_FIXTURE.md
    assert result["total_terms"] == 5
    term_freqs = result["term_frequencies"]

    # Assert counts
    assert term_freqs["alpha"] == 6
    assert term_freqs["always"] == 7
    assert term_freqs["beta"] == 3
    assert term_freqs["delta"] == 4
    assert term_freqs["gamma"] == 7

    # Assert exact key order (count descending, tie-broken by term name alphabetically ascending)
    expected_key_order = ["always", "gamma", "alpha", "delta", "beta"]
    assert list(term_freqs.keys()) == expected_key_order


def test_dtm_heatmap_tool():
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
    build_dtm_tool = next(t for t in tools if t.name == "build_dtm_tool")
    dtm_heatmap_tool = next(t for t in tools if t.name == "dtm_heatmap_tool")

    # Build DTM first
    build_dtm_tool.invoke({})

    # 1. Test with default parameters (n_terms=20, n_documents=20) -> returns full 8x5 matrix
    result_full = dtm_heatmap_tool.invoke({})
    assert "matrix_slice" in result_full
    assert result_full["term_names"] == ["alpha", "always", "beta", "delta", "gamma"]
    assert result_full["doc_indices"] == list(range(8))
    assert len(result_full["matrix_slice"]) == 8
    assert len(result_full["matrix_slice"][0]) == 5

    # Check pending_figure is set
    assert session.pending_figure is not None

    # 2. Test slice parameters n_terms=2, n_documents=3 -> returns [[3, 1], [2, 1], [1, 1]] (Known Answer #21)
    result_sliced = dtm_heatmap_tool.invoke({"n_terms": 2, "n_documents": 3})
    assert result_sliced["term_names"] == ["alpha", "always"]
    assert result_sliced["doc_indices"] == [0, 1, 2]
    assert result_sliced["matrix_slice"] == [[3, 1], [2, 1], [1, 1]]
