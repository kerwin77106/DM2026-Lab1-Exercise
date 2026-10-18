"""Tests for tools_exploration.py against agent_dev/sample_fixture.csv."""

import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_exploration import make_tools as make_exploration_tools


def test_cosine_similarity_tool():
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

    exp_tools = make_exploration_tools(session)
    cosine_similarity_tool = next(t for t in exp_tools if t.name == "cosine_similarity_tool")

    # Known Answer #16 in TEST_FIXTURE.md
    res_0_1 = cosine_similarity_tool.invoke({"doc_index_1": 0, "doc_index_2": 1})
    assert res_0_1["cosine_similarity"] == pytest.approx(0.9847, abs=1e-3)

    res_0_3 = cosine_similarity_tool.invoke({"doc_index_1": 0, "doc_index_2": 3})
    assert res_0_3["cosine_similarity"] == pytest.approx(0.0909, abs=1e-3)

    res_3_4 = cosine_similarity_tool.invoke({"doc_index_1": 3, "doc_index_2": 4})
    assert res_3_4["cosine_similarity"] == pytest.approx(0.9847, abs=1e-3)

    res_0_7 = cosine_similarity_tool.invoke({"doc_index_1": 0, "doc_index_2": 7})
    assert res_0_7["cosine_similarity"] == pytest.approx(0.0, abs=1e-3)


def test_feature_correlation_matrix_tool():
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

    exp_tools = make_exploration_tools(session)
    feature_corr_tool = next(t for t in exp_tools if t.name == "feature_correlation_matrix_tool")

    res = feature_corr_tool.invoke({})

    # Known Answer #19 in TEST_FIXTURE.md
    expected_terms = ["alpha", "gamma", "delta", "beta", "always"]
    assert res["terms"] == expected_terms

    corr_mat = np.array(res["correlation_matrix"])
    assert corr_mat.shape == (5, 5)

    # Check diagonal is all 1.0
    np.testing.assert_allclose(np.diag(corr_mat), np.ones(5), atol=1e-3)

    # Check alpha vs beta (index 0 vs index 3) = 0.8885
    assert corr_mat[0, 3] == pytest.approx(0.8885, abs=1e-3)

    # Check gamma vs delta (index 1 vs index 2) = 0.8307
    assert corr_mat[1, 2] == pytest.approx(0.8307, abs=1e-3)

    # Check figure set on pending_figure
    assert session.pending_figure is not None
