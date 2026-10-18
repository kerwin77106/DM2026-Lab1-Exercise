"""Tests for tools_patterns.py against agent_dev/sample_fixture.csv."""

import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_patterns import make_tools as make_patterns_tools


def test_mine_patterns_fpgrowth_tfidf():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })

    patterns_tools = make_patterns_tools(session)
    mine_patterns_tool = next(t for t in patterns_tools if t.name == "mine_patterns_tool")

    # 1. Test catA with filtering_method="tfidf", algorithm="fpgrowth", min_sup=1
    res_catA = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1,
    })

    assert "patterns" in res_catA
    assert len(res_catA["patterns"]) == 1
    assert res_catA["patterns"][0]["pattern"] == "alpha"
    assert res_catA["patterns"][0]["support"] == 3

    # 2. Test catB with filtering_method="tfidf", algorithm="fpgrowth", min_sup=1
    res_catB = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1,
    })

    assert "patterns" in res_catB
    assert len(res_catB["patterns"]) == 1
    assert res_catB["patterns"][0]["pattern"] == "gamma"
    assert res_catB["patterns"][0]["support"] == 4


def test_mine_patterns_zero_survived_filtering():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })

    patterns_tools = make_patterns_tools(session)
    mine_patterns_tool = next(t for t in patterns_tools if t.name == "mine_patterns_tool")

    # Known Answer #17: filtering_method="variance" keeps 0 terms for both categories
    res_var = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "variance",
        "algorithm": "fpgrowth",
        "min_sup": 1,
    })

    assert res_var["patterns"] == []
    assert "note" in res_var and res_var["note"] == "no terms survived filtering"


def test_mine_patterns_topk_and_maxfpgrowth():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })

    patterns_tools = make_patterns_tools(session)
    mine_patterns_tool = next(t for t in patterns_tools if t.name == "mine_patterns_tool")

    # Known Answer #20: algorithm="topk", k=1
    res_topk = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "k": 1,
    })

    assert len(res_topk["patterns"]) == 1
    assert res_topk["patterns"][0]["pattern"] == "alpha"
    assert res_topk["patterns"][0]["support"] == 3

    # Known Answer #20: algorithm="maxfpgrowth" returns 0 patterns for single-term database
    res_max = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "maxfpgrowth",
        "min_sup": 1,
    })

    assert res_max["patterns"] == []
