"""Tests for tools_reduction.py against agent_dev/sample_fixture.csv."""

import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_reduction import make_tools as make_reduction_tools


def test_reduce_dimensions_pca():
    session = SessionState()

    # Load dataset
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Build DTM
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Reduce dimensions via PCA
    reduction_tools = make_reduction_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    res = reduce_dimensions_tool.invoke({"method": "pca", "n_components": 2})

    # Assert shape and PCA variance ratios against Known Answer #7 in TEST_FIXTURE.md
    assert res["method"] == "pca"
    assert res["shape"] == [8, 2]
    assert res["explained_variance_ratio"] == pytest.approx([0.7532, 0.1965], abs=1e-3)

    # Check figure set on pending_figure
    assert session.pending_figure is not None

    # Check method-specific key in session.artifacts
    assert "reduced_coordinates_pca" in session.artifacts
    coords_pca = np.array(session.artifacts["reduced_coordinates_pca"])

    expected_pca = np.array([
        [2.3669, 0.8871],
        [1.7039, 0.2642],
        [1.0408, -0.3587],
        [-2.0759, 1.0129],
        [-1.4554, 0.3405],
        [-0.8349, -0.3319],
        [-0.8349, -0.3319],
        [0.0894, -1.4822],
    ])
    np.testing.assert_allclose(coords_pca, expected_pca, atol=1e-3)


def test_reduce_dimensions_tsne():
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

    reduction_tools = make_reduction_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    # Run 1: method="tsne", perplexity=2, random_state=42
    res1 = reduce_dimensions_tool.invoke({"method": "tsne", "perplexity": 2, "random_state": 42})

    assert res1["method"] == "tsne"
    assert res1["shape"] == [8, 2]
    assert "reduced_coordinates_tsne" in session.artifacts

    coords_tsne_1 = np.array(session.artifacts["reduced_coordinates_tsne"])

    # Assert no NaN or Inf in coordinates
    assert not np.isnan(coords_tsne_1).any()
    assert not np.isinf(coords_tsne_1).any()

    # Run 2: Repeat with same random_state=42 to verify deterministic repeatability
    res2 = reduce_dimensions_tool.invoke({"method": "tsne", "perplexity": 2, "random_state": 42})
    coords_tsne_2 = np.array(session.artifacts["reduced_coordinates_tsne"])

    # Assert exact repeatability when random_state is identical
    np.testing.assert_array_equal(coords_tsne_1, coords_tsne_2)


def test_reduce_dimensions_umap():
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

    reduction_tools = make_reduction_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    # Call with method="umap", n_neighbors=3, random_state=42 (Known Answer #9)
    res = reduce_dimensions_tool.invoke({"method": "umap", "n_neighbors": 3, "random_state": 42})

    assert res["method"] == "umap"
    assert res["shape"] == [8, 2]
    assert "reduced_coordinates_umap" in session.artifacts

    coords_umap = np.array(session.artifacts["reduced_coordinates_umap"])
    expected_umap = np.array([
        [6.6811, -6.8218],
        [6.9907, -7.2245],
        [7.7022, -7.3712],
        [10.2228, -6.2310],
        [9.8759, -5.6537],
        [9.3831, -6.5748],
        [9.2153, -5.8297],
        [8.5860, -7.2622],
    ])
    np.testing.assert_allclose(coords_umap, expected_umap, atol=1e-1)


def test_binarize_labels_tool():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })

    reduction_tools = make_reduction_tools(session)
    binarize_labels_tool = next(t for t in reduction_tools if t.name == "binarize_labels_tool")

    res = binarize_labels_tool.invoke({})

    assert "categories" in res
    assert sorted(res["categories"]) == ["catA", "catB"]
    assert res["shape"] == [8, 2]

    # Check artifacts and exact binary assertions
    assert "binarized_labels" in session.artifacts
    bin_df = session.artifacts["binarized_labels"]
    assert bin_df.shape == (8, 2)

    # 1. Assert every row sums to 1 (one-hot encoding)
    row_sums = bin_df.sum(axis=1).values
    np.testing.assert_array_equal(row_sums, np.ones(8, dtype=int))

    # 2. Assert row 0 is catA (catA=1, catB=0)
    assert bin_df.loc[0, "catA"] == 1
    assert bin_df.loc[0, "catB"] == 0

    # 3. Assert rows 3 to 6 are catB (catA=0, catB=1)
    for row_idx in range(3, 7):
        assert bin_df.loc[row_idx, "catA"] == 0
        assert bin_df.loc[row_idx, "catB"] == 1
