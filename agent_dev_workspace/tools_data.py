"""Tools for data loading, inspection, cleaning, and descriptive statistics."""

import os
from typing import Dict, Any, List, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from langchain_core.tools import tool


def make_tools(session: Any) -> list:
    """Creates and returns all data management tools bound to the given session."""

    @tool
    def list_files_tool(subdirectory: str = ".") -> Dict[str, Any]:
        """Lists files and directories in the specified workspace subdirectory.

        Enforces path containment within repository root and caps directory/file listings to top 100 each.

        Args:
            subdirectory (str): Subdirectory path relative to repository root (default ".").

        Returns:
            Dict[str, Any]: Summary containing result_id, subdirectory, n_directories, n_files,
            directories (up to 100), and files (up to 100).
        """
        repo_root = os.path.realpath(os.getcwd())
        target_path = os.path.realpath(os.path.join(repo_root, subdirectory))

        # 1. Path containment check
        if not (target_path == repo_root or target_path.startswith(repo_root + os.sep)):
            return {"error": f"Access denied: Subdirectory '{subdirectory}' is outside the repository root."}

        if not os.path.exists(target_path):
            return {"error": f"Subdirectory '{subdirectory}' does not exist."}

        if not os.path.isdir(target_path):
            return {"error": f"Path '{subdirectory}' is not a directory."}

        entries = os.listdir(target_path)
        all_directories = sorted([e for e in entries if os.path.isdir(os.path.join(target_path, e))])
        all_files = sorted([e for e in entries if os.path.isfile(os.path.join(target_path, e))])

        # 2. Cap lists to top 100 each while recording total counts
        n_directories = len(all_directories)
        n_files = len(all_files)

        directories_capped = all_directories[:100]
        files_capped = all_files[:100]

        result_id = session.next_result_id("list_files")
        summary = {
            "result_id": result_id,
            "subdirectory": subdirectory,
            "n_directories": n_directories,
            "n_files": n_files,
            "directories": directories_capped,
            "files": files_capped,
        }

        session.store_result("list_files_tool", {"subdirectory": subdirectory}, summary)
        return summary

    @tool
    def check_missing_tool() -> Dict[str, Any]:
        """Checks for missing, null, or empty text values in the loaded dataset.

        Returns:
            Dict[str, Any]: A summary containing 'result_id', 'missing_count',
            and 'missing_indices'.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded in session. Please load a dataset first."}

        if "text" not in session.dataframe.columns:
            return {"error": "'text' column not found in session dataframe."}

        # Identify missing/null values or empty/whitespace-only text strings
        text_series = session.dataframe["text"]
        is_null = text_series.isna()
        is_empty = text_series.astype(str).str.strip() == ""
        missing_mask = is_null | is_empty

        missing_indices = session.dataframe.index[missing_mask].tolist()
        missing_count = len(missing_indices)

        result_id = session.next_result_id("check_missing")
        summary = {
            "result_id": result_id,
            "missing_count": missing_count,
            "missing_indices": missing_indices,
            "total_documents": len(session.dataframe),
        }

        session.store_result("check_missing_tool", {}, summary)
        return summary

    @tool
    def check_duplicates_tool(drop: bool = False) -> Dict[str, Any]:
        """Checks for duplicate documents in the text column, and optionally drops them.

        When drop=True, removes all copies of duplicated text rows (keep=False).

        Args:
            drop (bool): If True, drops all copies of duplicate rows from session.dataframe.

        Returns:
            Dict[str, Any]: Summary containing 'result_id', 'duplicate_count',
            'duplicate_indices', 'dropped', and 'remaining_documents'.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded in session. Please load a dataset first."}

        if "text" not in session.dataframe.columns:
            return {"error": "'text' column not found in session dataframe."}

        # Identify duplicate rows (keep="first" counts extra duplicate copies)
        dup_mask_count = session.dataframe.duplicated(subset=["text"], keep="first")
        duplicate_indices = session.dataframe.index[dup_mask_count].tolist()
        duplicate_count = len(duplicate_indices)

        dropped = False
        if drop and duplicate_count > 0:
            # When dropping, drop ALL copies of duplicate rows (keep=False)
            session.dataframe = session.dataframe.drop_duplicates(
                subset=["text"], keep=False
            ).reset_index(drop=True)
            dropped = True

        result_id = session.next_result_id("check_duplicates")
        summary = {
            "result_id": result_id,
            "duplicate_count": duplicate_count,
            "duplicate_indices": duplicate_indices,
            "dropped": dropped,
            "remaining_documents": len(session.dataframe),
        }

        session.store_result("check_duplicates_tool", {"drop": drop}, summary)
        return summary

    @tool
    def inspect_data_tool(n: int = 5) -> Dict[str, Any]:
        """Inspects the first few rows of the dataset showing text and category_name columns.

        Truncates long text to the first 200 characters to keep summary small.

        Args:
            n (int): Number of rows to inspect (default 5).

        Returns:
            Dict[str, Any]: Summary containing 'result_id', 'total_rows', and 'head_records'.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded in session. Please load a dataset first."}

        cols = [c for c in ["text", "category_name"] if c in session.dataframe.columns]
        head_df = session.dataframe[cols].head(n).copy()

        if "text" in head_df.columns:
            head_df["text"] = head_df["text"].astype(str).apply(
                lambda t: t[:200] + "..." if len(t) > 200 else t
            )

        head_records = head_df.to_dict(orient="records")

        result_id = session.next_result_id("inspect_data")
        summary = {
            "result_id": result_id,
            "total_rows": len(session.dataframe),
            "head_records": head_records,
        }

        session.store_result("inspect_data_tool", {"n": n}, summary)
        return summary

    @tool
    def describe_data_tool() -> Dict[str, Any]:
        """Calculates text length summary statistics overall and per category, and plots box plot.

        Sets the generated boxplot on session.pending_figure.

        Returns:
            Dict[str, Any]: Summary containing 'result_id', 'overall_stats',
            and 'per_category_stats'.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded in session. Please load a dataset first."}

        if "text" not in session.dataframe.columns:
            return {"error": "'text' column not found in session dataframe."}

        # Calculate character length of each document
        df = session.dataframe.copy()
        df["text_length"] = df["text"].astype(str).apply(len)

        # Overall summary statistics
        overall_desc = df["text_length"].describe().to_dict()

        # Per category summary statistics
        cat_col = "category_name" if "category_name" in df.columns else ("category" if "category" in df.columns else None)
        per_category_desc = {}
        if cat_col:
            grouped = df.groupby(cat_col)["text_length"].describe()
            per_category_desc = grouped.to_dict(orient="index")

        # Create boxplot figure for session.pending_figure
        fig, ax = plt.subplots(figsize=(6, 4))
        if cat_col:
            sns.boxplot(data=df, x=cat_col, y="text_length", ax=ax)
            ax.set_xlabel("Category")
        else:
            sns.boxplot(data=df, y="text_length", ax=ax)
        ax.set_ylabel("Text Length (characters)")
        ax.set_title("Document Text Length Distribution")
        plt.tight_layout()

        # Set figure on session.pending_figure (no plt.show or fig.savefig)
        session.pending_figure = fig

        result_id = session.next_result_id("describe_data")
        summary = {
            "result_id": result_id,
            "overall_stats": overall_desc,
            "per_category_stats": per_category_desc,
        }

        session.store_result("describe_data_tool", {}, summary)
        return summary

    @tool
    def tokenize_tool() -> Dict[str, Any]:
        """Tokenizes text in session.dataframe into unigrams using nltk.word_tokenize.

        Stores list of token lists in session.dataframe['unigrams'] and session.artifacts['unigrams'].

        Returns:
            Dict[str, Any]: Summary containing result_id, total_documents, total_tokens,
            and tokens_head (first 5 tokenized documents, truncated to top 20 tokens each).
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded in session. Please load a dataset first."}

        if "text" not in session.dataframe.columns:
            return {"error": "'text' column not found in session dataframe."}

        # Ensure punkt is downloaded quietly if needed
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)

        def tokenize_text(val: Any) -> List[str]:
            if pd.isna(val) or not str(val).strip():
                return []
            return nltk.word_tokenize(str(val).lower())

        unigram_lists = session.dataframe["text"].apply(tokenize_text).tolist()

        # Store full unigram lists in dataframe and artifacts
        session.dataframe["unigrams"] = unigram_lists
        session.artifacts["unigrams"] = unigram_lists

        total_documents = len(unigram_lists)
        total_tokens = sum(len(tokens) for tokens in unigram_lists)

        # Truncate summary tokens_head to avoid huge JSON output in chat widget
        tokens_head = [tokens[:20] for tokens in unigram_lists[:5]]

        result_id = session.next_result_id("tokenize")
        summary = {
            "result_id": result_id,
            "total_documents": total_documents,
            "total_tokens": total_tokens,
            "tokens_head": tokens_head,
        }

        session.store_result("tokenize_tool", {}, summary)
        return summary

    @tool
    def sample_data_tool(
        n: int = 5,
        random_state: Optional[int] = 42,
    ) -> Dict[str, Any]:
        """Subsamples n random rows from session.dataframe.

        Stores the sampled DataFrame in session.artifacts['sampled_dataframe'].

        Args:
            n (int): Number of rows to sample (default 5).
            random_state (Optional[int]): Seed for random sampling (default 42).

        Returns:
            Dict[str, Any]: Summary containing result_id, n_sampled, sampled_indices, and sampled_head (first 5 sampled rows).
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded in session. Please load a dataset first."}

        total_rows = len(session.dataframe)
        actual_n = min(n, total_rows)

        # Sample rows
        sampled_df = session.dataframe.sample(n=actual_n, random_state=random_state)
        sampled_indices = sampled_df.index.tolist()

        # Store in session state artifacts
        session.artifacts["sampled_dataframe"] = sampled_df

        cols = [c for c in ["text", "category_name"] if c in sampled_df.columns]
        sampled_head_df = sampled_df[cols].head(5).copy()
        if "text" in sampled_head_df.columns:
            sampled_head_df["text"] = sampled_head_df["text"].astype(str).apply(
                lambda t: t[:200] + "..." if len(t) > 200 else t
            )

        sampled_head = sampled_head_df.to_dict(orient="records")

        result_id = session.next_result_id("sample_data")
        summary = {
            "result_id": result_id,
            "n_sampled": len(sampled_indices),
            "sampled_indices": sampled_indices,
            "sampled_head": sampled_head,
        }

        session.store_result("sample_data_tool", {"n": n, "random_state": random_state}, summary)
        return summary

    return [
        list_files_tool,
        check_missing_tool,
        check_duplicates_tool,
        inspect_data_tool,
        describe_data_tool,
        tokenize_tool,
        sample_data_tool,
    ]
