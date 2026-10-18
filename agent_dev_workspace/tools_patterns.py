"""Tools for frequent pattern mining using PAMI algorithms."""

import os
import tempfile
from typing import Dict, Any, Optional, Literal
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from langchain_core.tools import tool

# PAMI imports for pattern mining
import PAMI.extras.convert.DF2DB as db_converter
import PAMI.frequentPattern.basic.FPGrowth as fpgrowth
import PAMI.frequentPattern.topk.FAE as topk_fae
import PAMI.frequentPattern.maximal.MaxFPGrowth as maxfpgrowth


def make_tools(session: Any) -> list:
    """Creates and returns all pattern mining tools bound to the given session."""

    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: Literal["variance", "tfidf", "term_frequency"] = "tfidf",
        algorithm: Literal["fpgrowth", "topk", "maxfpgrowth"] = "fpgrowth",
        min_sup: int = 1,
        k: int = 10,
    ) -> Dict[str, Any]:
        """Mines frequent itemsets/word patterns for a single category using PAMI algorithms.

        Constructs a fresh category-local DTM (removing English stopwords), applies vocabulary filtering,
        converts to transactional DB matching Master's DF2DB syntax, and mines patterns.

        Args:
            category_name (str): Target category label string to mine.
            filtering_method (Literal["variance", "tfidf", "term_frequency"]): Vocabulary filtering method. Default "tfidf".
            algorithm (Literal["fpgrowth", "topk", "maxfpgrowth"]): Mining algorithm. Default "fpgrowth".
            min_sup (int): Minimum support count required for fpgrowth / maxfpgrowth. Default 1.
            k (int): Number of top-K patterns to keep for topk algorithm. Default 10.

        Returns:
            Dict[str, Any]: Summary containing result_id, category_name, filtering_method,
            algorithm, and top 20 patterns list by support descending.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded in session. Please load a dataset first."}

        # Identify category column
        cat_col = "category_name" if "category_name" in session.dataframe.columns else ("category" if "category" in session.dataframe.columns else None)
        if cat_col is None:
            return {"error": "Category column not found in session dataframe."}

        # 1. Filter dataframe to target category_name
        cat_df = session.dataframe[session.dataframe[cat_col] == category_name]
        if len(cat_df) == 0:
            return {"error": f"Category '{category_name}' not found in dataset."}

        # 2. Build fresh category-local DTM with stop_words='english'
        vectorizer = CountVectorizer(stop_words="english")
        texts = cat_df["text"].fillna("").astype(str)
        try:
            X_cat = vectorizer.fit_transform(texts)
            feature_names = list(vectorizer.get_feature_names_out())
        except ValueError:
            return {
                "result_id": session.next_result_id("mine_patterns"),
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "note": "no terms survived stopword removal",
            }

        if len(feature_names) == 0:
            return {
                "result_id": session.next_result_id("mine_patterns"),
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "note": "no terms survived stopword removal",
            }

        X_dense = X_cat.toarray()

        # 3. Apply chosen filtering method
        if filtering_method == "variance":
            scores = np.var(X_dense, axis=0)
            low, high = np.percentile(scores, [5, 95])
            surviving_mask = (scores >= low) & (scores <= high) & (scores > 0)
        elif filtering_method == "term_frequency":
            scores = np.sum(X_dense, axis=0)
            low, high = np.percentile(scores, [5, 95])
            surviving_mask = (scores >= low) & (scores <= high) & (scores > 0)
        elif filtering_method == "tfidf":
            transformer = TfidfTransformer()
            tfidf_matrix = transformer.fit_transform(X_cat).toarray()
            scores = np.mean(tfidf_matrix, axis=0)
            cutoff = np.percentile(scores, 20)
            # At or above the 20th percentile
            surviving_mask = scores >= cutoff
        else:
            return {"error": f"Unsupported filtering method '{filtering_method}'."}

        surviving_indices = np.where(surviving_mask)[0]
        if len(surviving_indices) == 0:
            result_id = session.next_result_id("mine_patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "note": "no terms survived filtering",
            }
            session.store_result("mine_patterns_tool", {
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
            }, summary)
            return summary

        surviving_terms = [feature_names[i] for i in surviving_indices]
        X_surviving = X_dense[:, surviving_indices]

        # 4. Convert to transactional database via PAMI DF2DB matching Master syntax
        binary_df = pd.DataFrame((X_surviving >= 1).astype(int), columns=surviving_terms)

        with tempfile.TemporaryDirectory() as tmpdir:
            db_file = os.path.join(tmpdir, "transactional.db")
            output_file = os.path.join(tmpdir, "patterns.txt")

            # Master syntax: obj = db_converter.DF2DB(binary_df); obj.convert2TransactionalDatabase(db_file, '>=', 1)
            obj = db_converter.DF2DB(binary_df)
            obj.convert2TransactionalDatabase(db_file, ">=", 1)

            # 5. Run mining algorithm
            if algorithm == "fpgrowth":
                miner = fpgrowth.FPGrowth(iFile=db_file, minSup=min_sup, sep="\t")
                miner.startMine()
                miner.save(output_file)
            elif algorithm == "topk":
                miner = topk_fae.FAE(iFile=db_file, k=k, sep="\t")
                miner.startMine()
                miner.save(output_file)
            elif algorithm == "maxfpgrowth":
                miner = maxfpgrowth.MaxFPGrowth(iFile=db_file, minSup=min_sup, sep="\t")
                miner.startMine()
                miner.save(output_file)
            else:
                return {"error": f"Unsupported algorithm '{algorithm}'."}

            # Parse patterns output file
            all_patterns = []
            if os.path.exists(output_file):
                with open(output_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split(":")
                        if len(parts) == 2:
                            pattern_str = parts[0].strip().replace("\t", " ")
                            try:
                                support_val = int(parts[1].strip())
                            except ValueError:
                                try:
                                    support_val = float(parts[1].strip())
                                except ValueError:
                                    support_val = parts[1].strip()
                            all_patterns.append({"pattern": pattern_str, "support": support_val})

        # Sort patterns by support descending and take top 20 for summary
        sorted_patterns = sorted(all_patterns, key=lambda x: x["support"] if isinstance(x["support"], (int, float)) else 0, reverse=True)
        top_20_patterns = sorted_patterns[:20]

        # Store all patterns in session.artifacts
        artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"
        session.artifacts[artifact_key] = all_patterns

        result_id = session.next_result_id("mine_patterns")
        summary = {
            "result_id": result_id,
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "patterns": top_20_patterns,
        }

        session.store_result("mine_patterns_tool", {
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "min_sup": min_sup,
            "k": k,
        }, summary)

        return summary

    return [mine_patterns_tool]
