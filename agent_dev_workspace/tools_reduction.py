"""Tools for dimensionality reduction and label binarization."""

from typing import Dict, Any, List, Optional, Literal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from langchain_core.tools import tool


def make_tools(session: Any) -> list:
    """Creates and returns all dimensionality reduction tools bound to the given session."""

    @tool
    def reduce_dimensions_tool(
        method: Literal["pca", "tsne", "umap"] = "pca",
        n_components: int = 2,
        perplexity: float = 30.0,
        n_neighbors: int = 15,
        random_state: Optional[int] = 42,
    ) -> Dict[str, Any]:
        """Reduces feature_matrix to 2D using PCA, t-SNE, or UMAP and plots a category scatter plot.

        Stores coordinates array in method-specific key e.g. session.artifacts['reduced_coordinates_pca'].
        Also maintains latest run at session.artifacts['reduced_coordinates'].
        Sets generated scatter plot figure on session.pending_figure.

        Args:
            method (Literal["pca", "tsne", "umap"]): Reduction technique to use. Default "pca".
            n_components (int): Target dimensions (default 2).
            perplexity (float): Perplexity parameter for t-SNE (default 30.0).
            n_neighbors (int): Number of neighbors parameter for UMAP (default 15).
            random_state (Optional[int]): Random seed for reproducibility (default 42).

        Returns:
            Dict[str, Any]: Summary containing result_id, method, shape, coordinates_head (first 5 points),
            and explained_variance_ratio (for PCA).
        """
        if session.feature_matrix is None:
            return {"error": "Document-term matrix not found. Please run build_dtm_tool first."}

        X = session.feature_matrix.toarray() if hasattr(session.feature_matrix, "toarray") else np.asarray(session.feature_matrix)

        explained_variance_ratio: Optional[List[float]] = None

        if method == "pca":
            reducer = PCA(n_components=n_components, random_state=random_state)
            coords_arr = reducer.fit_transform(X)
            explained_variance_ratio = [float(v) for v in reducer.explained_variance_ratio_]
        elif method == "tsne":
            reducer = TSNE(n_components=n_components, perplexity=perplexity, init="random", random_state=random_state)
            coords_arr = reducer.fit_transform(X)
        elif method == "umap":
            reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, random_state=random_state)
            coords_arr = reducer.fit_transform(X)
        else:
            return {"error": f"Unsupported method '{method}'. Choose from 'pca', 'tsne', or 'umap'."}

        # Convert coordinates array to python list of lists
        coords_list = coords_arr.tolist()

        # Store in method-specific key to avoid overwriting earlier runs when comparing methods
        session.artifacts[f"reduced_coordinates_{method}"] = coords_list
        # Also store under general key for latest run
        session.artifacts["reduced_coordinates"] = coords_list

        # Plot 2D scatter plot colored by category on session.pending_figure
        fig, ax = plt.subplots(figsize=(7, 5))
        labels = None
        if session.dataframe is not None:
            if "category_name" in session.dataframe.columns:
                labels = session.dataframe["category_name"].tolist()
            elif "category" in session.dataframe.columns:
                labels = session.dataframe["category"].tolist()

        plot_df = pd.DataFrame({
            "Dim1": coords_arr[:, 0],
            "Dim2": coords_arr[:, 1] if n_components > 1 else np.zeros(len(coords_arr)),
        })

        if labels is not None and len(labels) == len(coords_arr):
            plot_df["Category"] = labels
            sns.scatterplot(data=plot_df, x="Dim1", y="Dim2", hue="Category", ax=ax, palette="Set2", s=70)
        else:
            sns.scatterplot(data=plot_df, x="Dim1", y="Dim2", ax=ax, s=70)

        ax.set_title(f"2D Dimensionality Reduction ({method.upper()})")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        plt.tight_layout()

        session.pending_figure = fig

        # Prepare summary with shape and first 5 points (coordinates_head)
        result_id = session.next_result_id("reduce_dimensions")
        coords_head = [[round(val, 4) for val in point] for point in coords_list[:5]]

        summary = {
            "result_id": result_id,
            "method": method,
            "shape": list(coords_arr.shape),
            "coordinates_head": coords_head,
        }
        if explained_variance_ratio is not None:
            summary["explained_variance_ratio"] = [round(v, 4) for v in explained_variance_ratio]

        session.store_result("reduce_dimensions_tool", {
            "method": method,
            "n_components": n_components,
            "perplexity": perplexity,
            "n_neighbors": n_neighbors,
            "random_state": random_state,
        }, summary)

        return summary

    @tool
    def binarize_labels_tool() -> Dict[str, Any]:
        """One-hot encodes category labels in session.dataframe into a binary matrix.

        Stores binary DataFrame in session.artifacts['binarized_labels'].

        Returns:
            Dict[str, Any]: Summary containing result_id, categories, shape, and head_records.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded in session. Please load a dataset first."}

        col = None
        if "category_name" in session.dataframe.columns:
            col = "category_name"
        elif "category" in session.dataframe.columns:
            col = "category"

        if col is None:
            return {"error": "Category column not found in session dataframe."}

        # Perform one-hot encoding
        binarized_df = pd.get_dummies(session.dataframe[col], dtype=int)
        categories = list(binarized_df.columns)

        # Store in session state artifacts
        session.artifacts["binarized_labels"] = binarized_df

        head_records = binarized_df.head(5).to_dict(orient="records")

        result_id = session.next_result_id("binarize_labels")
        summary = {
            "result_id": result_id,
            "categories": categories,
            "shape": list(binarized_df.shape),
            "head_records": head_records,
        }

        session.store_result("binarize_labels_tool", {}, summary)
        return summary

    return [reduce_dimensions_tool, binarize_labels_tool]
