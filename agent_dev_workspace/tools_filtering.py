"""Tools for feature selection and correlation-based text filtering."""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from scipy.stats import rankdata
from langchain_core.tools import tool


def make_tools(session: Any) -> list:
    """Creates and returns all feature filtering tools bound to the given session."""

    @tool
    def variance_filter_tool(threshold: float = 0.0) -> Dict[str, Any]:
        """Calculates variance across all documents for each term in global DTM and filters low-variance terms.

        Computes variance sparse-matrix efficiently as E[X^2] - (E[X])^2.

        Args:
            threshold (float): Variance threshold below or equal to which terms are removed. Default is 0.0.

        Returns:
            Dict[str, Any]: Summary containing result_id, threshold, n_kept, n_removed,
            and variances (top 20 terms by variance descending).
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "Document-term matrix not found. Please run build_dtm_tool first."}

        # Sparse variance computation: Var = E[X^2] - (E[X])^2 without dense .toarray()
        X = session.feature_matrix
        mean = np.asarray(X.mean(axis=0)).flatten()
        mean_sq = np.asarray(X.power(2).mean(axis=0)).flatten()
        variances_arr = np.maximum(0.0, mean_sq - mean ** 2)

        # Map terms to variance values
        term_variances = {term: float(var) for term, var in zip(session.feature_names, variances_arr)}

        # Filter terms by threshold
        kept_features = [term for term, var in term_variances.items() if var > threshold]
        removed_features = [term for term, var in term_variances.items() if var <= threshold]

        # Prepare full_report DataFrame for visualize_result_tool (2 columns, sorted descending)
        df_report = pd.DataFrame([
            {"feature": term, "variance": var}
            for term, var in term_variances.items()
        ]).sort_values(by="variance", ascending=False).reset_index(drop=True)

        # Top 20 variances dict for summary
        sorted_items = sorted(term_variances.items(), key=lambda x: x[1], reverse=True)[:20]
        summary_variances = {term: round(var, 4) for term, var in sorted_items}

        result_id = session.next_result_id("variance")
        summary = {
            "result_id": result_id,
            "threshold": threshold,
            "n_kept": len(kept_features),
            "n_removed": len(removed_features),
            "variances": summary_variances,
        }

        # Pass full_report DataFrame as full_report arg to session.store_result
        session.store_result("variance_filter_tool", {"threshold": threshold}, summary, full_report=df_report)

        return summary

    @tool
    def pearson_filter_tool(
        target_class: str,
        threshold: float = 0.0,
    ) -> Dict[str, Any]:
        """Calculates Pearson correlation between each term count and a binary target category indicator.

        Computes all term correlations simultaneously on the sparse matrix without dense conversion.

        Args:
            target_class (str): Target category label string to compute one-vs-rest correlation against.
            threshold (float): Minimum absolute Pearson correlation value |r| required to keep a term. Default 0.0.

        Returns:
            Dict[str, Any]: Summary containing result_id, target_class, threshold, n_kept, n_removed,
            and correlations (top 20 terms by absolute Pearson correlation descending).
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "Document-term matrix not found. Please run build_dtm_tool first."}

        if session.dataframe is None:
            return {"error": "No dataset loaded in session. Please load a dataset first."}

        # Extract target binary indicator vector y (1 if target_class else 0)
        labels = None
        if "category_name" in session.dataframe.columns:
            labels = session.dataframe["category_name"].values
        elif "category" in session.dataframe.columns:
            labels = session.dataframe["category"].values

        if labels is None:
            return {"error": "Category labels not found in session dataframe."}

        y = (labels == target_class).astype(float)

        if np.all(y == 0):
            return {"error": f"Target class '{target_class}' not found in dataset labels."}

        # Sparse-matrix parallel vector computation for Pearson correlation across all terms
        X = session.feature_matrix
        N = X.shape[0]

        mean_x = np.asarray(X.mean(axis=0)).flatten()
        mean_y = float(y.mean())

        std_y = float(np.std(y))

        mean_sq_x = np.asarray(X.power(2).mean(axis=0)).flatten()
        var_x = np.maximum(0.0, mean_sq_x - mean_x ** 2)
        std_x = np.sqrt(var_x)

        # E[X * y] via sparse matrix-vector multiplication
        mean_xy = np.asarray(X.T.dot(y)).flatten() / N

        cov_xy = mean_xy - mean_x * mean_y

        # Handle zero-std columns safely
        valid_mask = (std_x > 0) & (std_y > 0)
        r_arr = np.zeros_like(std_x, dtype=float)
        r_arr[valid_mask] = cov_xy[valid_mask] / (std_x[valid_mask] * std_y)
        r_arr = np.nan_to_num(r_arr, nan=0.0)

        # Map terms to Pearson r values
        correlations = {term: float(r) for term, r in zip(session.feature_names, r_arr)}

        # Filter features based on absolute correlation threshold |r| > threshold
        kept_features = [term for term, r in correlations.items() if abs(r) > threshold]
        removed_features = [term for term, r in correlations.items() if abs(r) <= threshold]

        # Prepare full_report DataFrame for visualize_result_tool (feature, pearson_r)
        df_report = pd.DataFrame([
            {"feature": term, "pearson_r": r}
            for term, r in correlations.items()
        ]).sort_values(by="pearson_r", key=abs, ascending=False).reset_index(drop=True)

        # Top 20 terms by absolute correlation for summary
        sorted_items = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:20]
        summary_correlations = {term: round(r, 4) for term, r in sorted_items}

        result_id = session.next_result_id("pearson")
        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "threshold": threshold,
            "n_kept": len(kept_features),
            "n_removed": len(removed_features),
            "correlations": summary_correlations,
        }

        session.store_result(
            "pearson_filter_tool",
            {"target_class": target_class, "threshold": threshold},
            summary,
            full_report=df_report,
        )

        return summary

    @tool
    def spearman_filter_tool(
        target_class: str,
        threshold: float = 0.0,
    ) -> Dict[str, Any]:
        """Calculates Spearman rank correlation between each term count and a binary target category indicator.

        Ranks columns of the DTM and target indicator, then calculates correlation.

        Args:
            target_class (str): Target category label string to compute correlation against.
            threshold (float): Minimum absolute Spearman correlation value |r| required to keep a term. Default 0.0.

        Returns:
            Dict[str, Any]: Summary containing result_id, target_class, threshold, n_kept, n_removed,
            and correlations (top 20 terms by absolute Spearman correlation descending).
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "Document-term matrix not found. Please run build_dtm_tool first."}

        if session.dataframe is None:
            return {"error": "No dataset loaded in session. Please load a dataset first."}

        # Extract target binary indicator vector y (1 if target_class else 0)
        labels = None
        if "category_name" in session.dataframe.columns:
            labels = session.dataframe["category_name"].values
        elif "category" in session.dataframe.columns:
            labels = session.dataframe["category"].values

        if labels is None:
            return {"error": "Category labels not found in session dataframe."}

        y = (labels == target_class).astype(float)

        if np.all(y == 0):
            return {"error": f"Target class '{target_class}' not found in dataset labels."}

        # Convert feature matrix to dense array for rank computation
        X_dense = session.feature_matrix.toarray() if hasattr(session.feature_matrix, "toarray") else np.asarray(session.feature_matrix)

        # Compute ranks along columns (axis=0) for X, and rank for vector y
        X_ranked = rankdata(X_dense, axis=0)
        y_ranked = rankdata(y)

        # Compute Pearson correlation on ranked values (= Spearman correlation)
        N = X_ranked.shape[0]
        mean_x = X_ranked.mean(axis=0)
        mean_y = float(y_ranked.mean())

        std_y = float(np.std(y_ranked))

        std_x = np.std(X_ranked, axis=0)

        mean_xy = np.dot(X_ranked.T, y_ranked) / N
        cov_xy = mean_xy - mean_x * mean_y

        valid_mask = (std_x > 0) & (std_y > 0)
        rho_arr = np.zeros_like(std_x, dtype=float)
        rho_arr[valid_mask] = cov_xy[valid_mask] / (std_x[valid_mask] * std_y)
        rho_arr = np.nan_to_num(rho_arr, nan=0.0)

        # Map terms to Spearman rho values
        correlations = {term: float(rho) for term, rho in zip(session.feature_names, rho_arr)}

        # Filter features based on absolute correlation threshold |rho| > threshold
        kept_features = [term for term, rho in correlations.items() if abs(rho) > threshold]
        removed_features = [term for term, rho in correlations.items() if abs(rho) <= threshold]

        # Prepare full_report DataFrame for visualize_result_tool (feature, spearman_r)
        df_report = pd.DataFrame([
            {"feature": term, "spearman_r": rho}
            for term, rho in correlations.items()
        ]).sort_values(by="spearman_r", key=abs, ascending=False).reset_index(drop=True)

        # Top 20 terms by absolute correlation for summary
        sorted_items = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:20]
        summary_correlations = {term: round(rho, 4) for term, r in sorted_items for rho in [r]}

        result_id = session.next_result_id("spearman")
        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "threshold": threshold,
            "n_kept": len(kept_features),
            "n_removed": len(removed_features),
            "correlations": summary_correlations,
        }

        session.store_result(
            "spearman_filter_tool",
            {"target_class": target_class, "threshold": threshold},
            summary,
            full_report=df_report,
        )

        return summary

    return [variance_filter_tool, pearson_filter_tool, spearman_filter_tool]
