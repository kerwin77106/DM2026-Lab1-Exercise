"""Tools for dataset exploration and document/feature correlation analysis."""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.tools import tool


def make_tools(session: Any) -> list:
    """Creates and returns all data exploration tools bound to the given session."""

    @tool
    def cosine_similarity_tool(
        doc_index_1: int,
        doc_index_2: int,
    ) -> Dict[str, Any]:
        """Calculates cosine similarity between two documents in session.dataframe using DTM vectors or count_vectorizer.

        Args:
            doc_index_1 (int): Index of first document.
            doc_index_2 (int): Index of second document.

        Returns:
            Dict[str, Any]: Summary containing result_id, doc_index_1, doc_index_2,
            cosine_similarity, and text snippets.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded in session. Please load a dataset first."}

        total_docs = len(session.dataframe)
        if doc_index_1 < 0 or doc_index_1 >= total_docs:
            return {"error": f"doc_index_1 ({doc_index_1}) out of range (0-{total_docs - 1})."}
        if doc_index_2 < 0 or doc_index_2 >= total_docs:
            return {"error": f"doc_index_2 ({doc_index_2}) out of range (0-{total_docs - 1})."}

        # Extract text strings
        text1 = str(session.dataframe.iloc[doc_index_1]["text"]) if "text" in session.dataframe.columns else ""
        text2 = str(session.dataframe.iloc[doc_index_2]["text"]) if "text" in session.dataframe.columns else ""

        # Option 1: Use session.artifacts["count_vectorizer"] if available
        if "count_vectorizer" in session.artifacts and session.artifacts["count_vectorizer"] is not None:
            vectorizer = session.artifacts["count_vectorizer"]
            v1 = vectorizer.transform([text1])
            v2 = vectorizer.transform([text2])
        # Option 2: Fallback to slicing session.feature_matrix directly
        elif session.feature_matrix is not None:
            v1 = session.feature_matrix[doc_index_1]
            v2 = session.feature_matrix[doc_index_2]
        else:
            return {"error": "Document-term matrix not found. Please run build_dtm_tool first."}

        norm1 = float(np.linalg.norm(v1.toarray() if hasattr(v1, "toarray") else v1))
        norm2 = float(np.linalg.norm(v2.toarray() if hasattr(v2, "toarray") else v2))

        # Safe zero-norm handling (e.g. empty text rows)
        if norm1 == 0.0 or norm2 == 0.0:
            sim_score = 0.0
        else:
            sim_matrix = cosine_similarity(v1, v2)
            sim_score = float(sim_matrix[0, 0])

        snippet1 = text1[:100] + "..." if len(text1) > 100 else text1
        snippet2 = text2[:100] + "..." if len(text2) > 100 else text2

        result_id = session.next_result_id("cosine_sim")
        summary = {
            "result_id": result_id,
            "doc_index_1": doc_index_1,
            "doc_index_2": doc_index_2,
            "cosine_similarity": round(sim_score, 4),
            "doc_1_snippet": snippet1,
            "doc_2_snippet": snippet2,
        }

        session.store_result("cosine_similarity_tool", {
            "doc_index_1": doc_index_1,
            "doc_index_2": doc_index_2,
        }, summary)

        return summary

    @tool
    def feature_correlation_matrix_tool() -> Dict[str, Any]:
        """Calculates term-vs-term correlation matrix over top 20 terms by variance in global DTM.

        Plots annotated heatmap figure on session.pending_figure.

        Returns:
            Dict[str, Any]: Summary containing result_id, terms, and correlation_matrix.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "Document-term matrix not found. Please run build_dtm_tool first."}

        X = session.feature_matrix.toarray() if hasattr(session.feature_matrix, "toarray") else np.asarray(session.feature_matrix)

        # Compute variance across documents for each term
        variances = np.var(X, axis=0)

        # Sort terms by variance descending and take top 20
        top_indices = np.argsort(-variances)[:20]
        top_terms = [session.feature_names[i] for i in top_indices]
        X_top = X[:, top_indices]

        # Compute Pearson correlation matrix across top terms
        corr_mat = np.corrcoef(X_top, rowvar=False)
        corr_mat = np.nan_to_num(corr_mat, nan=0.0)

        # Plot annotated correlation matrix heatmap figure on session.pending_figure
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            corr_mat,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            xticklabels=top_terms,
            yticklabels=top_terms,
            ax=ax,
            vmin=-1.0,
            vmax=1.0,
        )
        ax.set_title("Feature Correlation Matrix (Top Variance Terms)")
        plt.tight_layout()

        session.pending_figure = fig

        # Round correlation matrix entries for clean JSON output
        corr_list = [[round(float(val), 4) for val in row] for row in corr_mat]

        result_id = session.next_result_id("feature_corr")
        summary = {
            "result_id": result_id,
            "terms": top_terms,
            "correlation_matrix": corr_list,
        }

        session.store_result("feature_correlation_matrix_tool", {}, summary)
        return summary

    return [cosine_similarity_tool, feature_correlation_matrix_tool]
