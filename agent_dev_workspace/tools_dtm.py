"""Tools for Document-Term Matrix (DTM) construction and term frequency analysis."""

from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from langchain_core.tools import tool


def make_tools(session: Any) -> list:
    """Creates and returns all DTM tools bound to the given session."""

    @tool
    def build_dtm_tool(
        max_features: Optional[int] = None,
        min_df: Union[int, float] = 1,
        max_df: Union[int, float] = 1.0,
        ngram_range: List[int] = [1, 1],
    ) -> Dict[str, Any]:
        """Builds a Document-Term Matrix (DTM) from the text column in session.dataframe.

        Stores vectorizer in session.artifacts['count_vectorizer'], DTM matrix in
        session.feature_matrix, and feature names in session.feature_names.

        Args:
            max_features (Optional[int]): Max vocabulary size.
            min_df (Union[int, float]): Minimum document frequency.
            max_df (Union[int, float]): Maximum document frequency.
            ngram_range (List[int]): Range of n-grams as [min_n, max_n] list.

        Returns:
            Dict[str, Any]: Summary containing result_id, vocabulary_size, non_zero,
            total_elements, sparsity_pct, and feature_names (first 20 terms).
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded in session. Please load a dataset first."}

        if "text" not in session.dataframe.columns:
            return {"error": "'text' column not found in session dataframe."}

        # Convert List[int] to tuple for CountVectorizer to be safe for Gemini backend schema
        ngram_tuple = tuple(ngram_range) if isinstance(ngram_range, list) else ngram_range

        # Fit CountVectorizer on session.dataframe['text']
        vectorizer = CountVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_tuple,
        )

        texts = session.dataframe["text"].fillna("").astype(str)
        feature_matrix = vectorizer.fit_transform(texts)
        feature_names = list(vectorizer.get_feature_names_out())

        # Store in session state
        session.feature_matrix = feature_matrix
        session.feature_names = feature_names
        session.artifacts["count_vectorizer"] = vectorizer

        # Set labels if category_name column exists
        if "category_name" in session.dataframe.columns:
            session.set_labels(session.dataframe["category_name"].values)
        elif "category" in session.dataframe.columns:
            session.set_labels(session.dataframe["category"].values)

        # Sparsity calculation
        n_rows, n_cols = feature_matrix.shape
        non_zero = int(feature_matrix.nnz)
        total_elements = int(n_rows * n_cols)
        sparsity_pct = float(100.0 * (1.0 - non_zero / total_elements)) if total_elements > 0 else 0.0

        result_id = session.next_result_id("build_dtm")
        summary = {
            "result_id": result_id,
            "vocabulary_size": len(feature_names),
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": round(sparsity_pct, 4),
            "feature_names": feature_names[:20],  # Truncate summary to first 20 terms
        }

        session.store_result("build_dtm_tool", {
            "max_features": max_features,
            "min_df": min_df,
            "max_df": max_df,
            "ngram_range": ngram_range,
        }, summary)

        return summary

    @tool
    def term_frequency_tool() -> Dict[str, Any]:
        """Aggregates total frequencies across all documents for each term in the global DTM.

        Returns:
            Dict[str, Any]: Summary containing result_id, total_terms, and term_frequencies
            (up to top 20 terms sorted by count descending, then term name alphabetically).
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "Document-term matrix not found. Please run build_dtm_tool first."}

        # Calculate column sums (total count per term)
        term_sums = np.asarray(session.feature_matrix.sum(axis=0)).flatten()

        # Map feature_names to Python int count
        raw_counts = {term: int(count) for term, count in zip(session.feature_names, term_sums)}

        # Sort descending by count, then ascending by term name for ties
        sorted_terms = sorted(raw_counts.items(), key=lambda item: (-item[1], item[0]))

        # Limit summary to top 20 terms
        top_20 = sorted_terms[:20]
        top_term_frequencies = {term: count for term, count in top_20}

        result_id = session.next_result_id("term_frequency")
        summary = {
            "result_id": result_id,
            "total_terms": len(raw_counts),
            "term_frequencies": top_term_frequencies,
        }

        session.store_result("term_frequency_tool", {}, summary)
        return summary

    @tool
    def dtm_heatmap_tool(
        n_terms: int = 20,
        n_documents: int = 20,
    ) -> Dict[str, Any]:
        """Plots a positional slice heatmap of the raw global Document-Term Matrix (DTM).

        Slices the DTM up to n_documents (rows) and n_terms (columns), sets heatmap figure on
        session.pending_figure, and returns matrix_slice with document and term labels.

        Args:
            n_terms (int): Maximum number of terms (columns) to include in slice (default 20).
            n_documents (int): Maximum number of documents (rows) to include in slice (default 20).

        Returns:
            Dict[str, Any]: Summary containing result_id, doc_indices, term_names, and matrix_slice.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "Document-term matrix not found. Please run build_dtm_tool first."}

        # Bound n_terms and n_documents to available matrix dimensions
        max_docs, max_terms = session.feature_matrix.shape
        actual_docs = min(n_documents, max_docs)
        actual_terms = min(n_terms, max_terms)

        # Slice submatrix and convert to dense array
        submatrix = session.feature_matrix[:actual_docs, :actual_terms]
        slice_array = submatrix.toarray() if hasattr(submatrix, "toarray") else np.asarray(submatrix)

        selected_terms = list(session.feature_names[:actual_terms])
        doc_indices = list(range(actual_docs))

        # Build heatmap figure
        fig, ax = plt.subplots(figsize=(max(6, actual_terms * 0.5), max(4, actual_docs * 0.4)))
        sns.heatmap(
            slice_array,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=selected_terms,
            yticklabels=[f"Doc {i}" for i in doc_indices],
            ax=ax,
            cbar=True,
        )
        ax.set_title(f"Document-Term Matrix Slice ({actual_docs} Docs x {actual_terms} Terms)")
        ax.set_xlabel("Terms")
        ax.set_ylabel("Document Index")
        plt.tight_layout()

        # Set pending figure for notebook rendering
        session.pending_figure = fig

        # Prepare summary dictionary
        matrix_slice = slice_array.tolist()
        result_id = session.next_result_id("dtm_heatmap")
        summary = {
            "result_id": result_id,
            "doc_indices": doc_indices,
            "term_names": selected_terms,
            "matrix_slice": matrix_slice,
        }

        session.store_result("dtm_heatmap_tool", {"n_terms": n_terms, "n_documents": n_documents}, summary)
        return summary

    return [build_dtm_tool, term_frequency_tool, dtm_heatmap_tool]
