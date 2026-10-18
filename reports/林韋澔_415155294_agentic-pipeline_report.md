# Your report
Note: the kernel was restarted after a mine_patterns_tool run used about 30 GB of memory. When I filled in the setup form again, I mistakenly typed "Restart Kernel" into the Student ID field, so the second session's log and the plots reduce_dimensions_4 to 6 carry that text instead of my student ID, 415155294. Result IDs from mine_patterns_3 onward come from this second session


## 1. Data and quality checks
The data is clean and needs no cleaning
total of 2,257 clean documents were loaded across four categories , with zero missing  or duplicate values . While comp.graphics has the shortest median length , it features the dataset's longest single post , likely due to embedded source code or raw data rather than standard discussion.

![Text length by category](../plots/林韋澔_agentic-pipeline_plot_describe_data_5.png)

## 2. Document-term matrix: stopwords dominate

DTM & Sparsity : Generated 35,788 terms with 99.547% sparsity because stopwords were retained .

Stopwords Dominance : 19 of the top 20 terms are stopwords , plus "edu" from headers. The DTM slice heatmap is overwhelmingly sparse .

Variance Filter Limitation: A 0.01 threshold keeps 5,354 terms. It fails to remove stopwords because their frequency fluctuates across documents, keeping their variance high.

Topical Exception : "jpeg" achieves high variance due to its concentration in comp.graphics.

![DTM slice heatmap](../plots/林韋澔_agentic-pipeline_plot_dtm_heatmap_8.png)

![Top terms by variance](../plots/林韋澔_agentic-pipeline_plot_turn98.png)
## 3. Correlation with comp.graphics

I ran Pearson and Spearman filters with comp.graphics as the target class. In `pearson_10`, the terms shared with Master Q6 have exactly the same values, for example "thanks" at 0.2778 and "graphics" at 0.1779, because Pearson is computed for each term separately. The difference is that this matrix includes more terms, and the stopwords that Master had removed now appear with negative correlations: "re" (-0.2566), "who", "that", "not" and "what". This makes sense because comp.graphics posts tend to be short technical questions or code, with fewer of the pronouns and negations that are common in the long discussions and arguments of the religion groups.

In `spearman_11`, "graphics" is again first at 0.5211, the same as Master Q6, but 10 of the top 20 terms are now negatively correlated stopwords, and they are much stronger than in Pearson: "that" is -0.3906 here versus -0.2077 in `pearson_10`. The reason is the extreme values in the data. Spearman uses ranks, so the very long posts found in `describe_data_5` do not distort it, and it captures the consistent pattern that comp.graphics posts are shorter and therefore contain fewer stopwords.

The feature correlation matrix (`feature_corr_12`) of the top 20 terms by variance supports this length explanation. The stopwords are strongly correlated with each other, from 0.53 up to 0.94 ("the" and "of"), because they all grow with the length of the message body. "edu" has the lowest correlations, down to 0.09, because it mainly appears in the email header rather than the body, so its count does not rise and fall with the other words. It works like a control: words that scale with body length move together, while a header word does not.

![Pearson correlation with comp.graphics](../plots/林韋澔_agentic-pipeline_plot_turn104.png)

![Spearman correlation with comp.graphics](../plots/林韋澔_agentic-pipeline_plot_turn110.png)

![Feature correlation matrix](../plots/林韋澔_agentic-pipeline_plot_feature_corr_12.png)
## 4. Pattern mining and dimensionality reduction


A first `mine_patterns_tool` run on comp.graphics with FP-Growth and the default `min_sup=1` used almost 30 GB of memory and had to be stopped, so the kernel was restarted; the result_ids below come from this second session. I then mined sci.med with TF-IDF filtering and Top-K (k=20), which fixes the number of patterns instead of guessing a support threshold. Unlike the global DTM, this tool removes English stopwords, but the email header fields are still there, so all 20 patterns in `mine_patterns_3` are header words: "subject" (support 594), "lines" (585), "organization" (582), "edu" (416), "writes" (344), "article" (343) and their combinations. "subject" has a support of 594, which is every sci.med post, because every message has a Subject line. Removing stopwords only let the header words take over, so isolating one category, as I argued in Master Q1, is not enough on its own; the headers also have to be stripped before pattern mining can show what sci.med is about.

PCA (`reduce_dimensions_4`) is dominated by one direction: PC1 explains 65.94% of the variance and PC2 only 4.33%, which is unusual for 35,788 dimensions. Because most high-variance terms are stopwords that rise and fall together, as shown in `feature_corr_12`, PC1 mainly captures document length rather than topic. Most documents overlap near the origin, and the four comp.graphics points at the top are the very long posts from `describe_data_5`, matching my Master Q9 answer that PCA is dominated by outliers.

t-SNE (`reduce_dimensions_5`) spreads the documents out into rough regions, but the two religion groups overlap heavily. From this plot I hypothesized that the layout follows document length from left to right, since comp.graphics was on the left and the longer religion posts on the right. UMAP (`reduce_dimensions_6`) does not support this: it shows clearer category clusters, with alt.atheism in its own region and an isolated sci.med island, but comp.graphics is on the right instead of the left. This suggests that different reduction methods differ in how sensitive they are to the length artifact versus real semantic structure, and that the left-to-right order in t-SNE should not be over-interpreted, as I noted in Master Q9.


![PCA](../plots/林韋澔_agentic-pipeline_plot_reduce_dimensions_4.png)

![t-SNE](../plots/林韋澔_agentic-pipeline_plot_reduce_dimensions_5.png)

![UMAP](../plots/林韋澔_agentic-pipeline_plot_reduce_dimensions_6.png)
## 5. Conclusion

The whole analysis shows that unfiltered stopwords and email header metadata dominate the results at almost every step: term frequency (`term_frequency_7`), variance filtering (`variance_9`), the correlation matrix (`feature_corr_12`), pattern mining (`mine_patterns_3`) and dimensionality reduction, where PC1 of PCA mainly reflects document length (`reduce_dimensions_4`). The last two tools point the same way. `binarize_labels_7` produced 4 one-hot columns for the 4 categories, the same as Master Q10, and `cosine_sim_8` gave a similarity of 0.3532 between two comp.graphics posts on unrelated subjects, a laser printer and a trimming region, which is probably inflated by the shared stopwords and header words rather than a shared topic; I did not run a cross-category pair to confirm this.

For a real topic analysis, the next step should be a thorough data cleaning before building the matrix: removing stopwords with a stopword list, and stripping the email headers and other irrelevant fixed fields such as From, Subject, Organization and Lines, either manually or with a parser.

Several results match the Master notebook exactly, which confirms that my tools are implemented correctly: the Pearson correlations of terms like "thanks" (0.2778) and "graphics" (0.1779), the Spearman value of "graphics" (0.5211), the 35,788 terms and 99.547% sparsity that match Master Q4 with `stop_words=None`, and the category counts, such as 594 sci.med documents as in Master Q1.