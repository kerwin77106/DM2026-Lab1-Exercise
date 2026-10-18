# Homework Report

## 1. Data: imbalanced and short

I loaded `newdataset/Reddit-stock-sentiment.csv` (`load_dataset_2`) after confirming the file with my own `list_files_tool` (`list_files_1`). It differs from 20-newsgroups in four ways. First, the labels are highly imbalanced: of 847 posts, 423 are neutral, 315 negative and only 109 positive, about 12.9%, while the four newsgroup categories were much more even (480 to 599 each). Second, the posts are very short: after building the matrix there are only about 20 distinct words per post, versus about 162 in the newsgroups. Third, there are no email headers. Fourth, the sentiment depends heavily on financial slang, which a bag-of-words model has trouble capturing: in `inspect_data_3`, "Seeing lots of red in the ticker" is labeled neutral, although "red" in stock talk means prices are falling.

There were no missing texts (`check_missing_4`), but `check_duplicates_5` found 23 duplicate records. I decided to remove them before building the document-term matrix, so that repeated text would not inflate term frequencies, variances and correlations. Because the tool uses `keep=False`, every copy is removed rather than keeping one, so 34 rows from 11 duplicated texts were dropped and 813 documents remained (`check_duplicates_6`). Almost all of them were neutral: neutral went from 423 to 391, positive from 109 to 107, and negative stayed at 315.

The text length distribution (`describe_data_7`) is right-skewed, like the 20-newsgroups one: most posts are very short and only a few are long. It also repeats the same pattern I saw for comp.graphics in the pipeline: the neutral class has the shortest median post, yet it also has the longest post, about 2,753 characters. The scale is very different, though: that longest post is about 22 times shorter than the longest newsgroup post, 60,713 characters.

![Text length by sentiment](../plots/林韋澔_homework_plot_describe_data_7.png)


## 2. Document-term matrix and filtering

DTM and Sparsity: Although the vocabulary size is only one-eighth of the pipeline's newsgroup dataset, the sparsity remains nearly identical (~99.5%) because the posts are extremely short (averaging about 20 distinct words per post).

Term Frequency: While top terms are heavily dominated by stopwords just like the pipeline, this dataset lacks domain markers (such as "edu") and instead features conversational pronouns (e.g., "he"), reflecting Reddit's community-driven discussion style.

Variance Filter: The variance filter retains fewer terms (11.4% versus 15.0%) because short, scattered posts prevent words from appearing frequently enough to reach the variance threshold, meaning no dominant topic words (like "jpeg") emerge at the top.

## 3. Correlation with the positive class

Pearson Correlation: Top correlations are much weaker (max ~0.156) and feature frequent ties because terms appearing in only a few posts (all positive) yield identical mathematical scores. With only 107 positive posts, these rankings are driven by chance rather than reliability, highlighting the risk of dataset imbalance.

Spearman vs. Pearson: Both metrics nearly coincide due to sparse 0/1 counts and minimal ranking changes, confirming that rank-based and value-based correlations only diverge when counts vary widely with outliers.

Feature Correlation Matrix: Maximum correlations are lower (~0.69 vs. 0.94 in the pipeline) because uniform short post lengths minimize document-length effects. Stronger pairs instead reflect common predictive phrases, such as "will be."

![Pearson correlation with the positive class](../plots/林韋澔_homework_plot_turn74.png)

![Feature correlation matrix](../plots/林韋澔_homework_plot_feature_corr_13.png)

## 4. Pattern mining and dimensionality reduction

Pattern Mining: Without email headers, pattern mining successfully uncovers meaningful single words (sentiment terms, market topics, and slang). However, it cannot find multi-word patterns due to short posts and limited positive samples (low support).

Dimensionality Reduction: Unlike the pipeline, PCA PC1 explains much less variance (26.70%) because Reddit posts have more uniform lengths. Meanwhile, t-SNE and UMAP fail entirely to separate the sentiment labels, showing complete mixing.

Core Limitation: The failure to cluster stems from the bag-of-words model and short texts. Because positive, neutral, and negative posts share nearly identical vocabulary (differing only in tone and context), the sparse vector space lacks the geometric separation required for dimensionality reduction.

![PCA](../plots/林韋澔_homework_plot_reduce_dimensions_15.png)

![t-SNE](../plots/林韋澔_homework_plot_reduce_dimensions_16.png)

![UMAP](../plots/林韋澔_homework_plot_reduce_dimensions_17.png)

## 5. Conclusion: comparison with the pipeline

This dataset confirmed three conclusions from my pipeline report. First, email headers: in the pipeline, pattern mining was dominated by header words like "subject"; Reddit has no headers, and pattern mining found real content, market topics and sentiment words, which shows the method was fine and the headers were the noise. Second, document length: in the pipeline I argued that the strong stopword correlations and the 65.94% PC1 were driven by very large differences in document length. Reddit posts are short and similar in length, and both effects dropped, to a strongest correlation of 0.6922 and a PC1 of 26.70%. Third, stopwords: raw frequencies and the top-variance terms are dominated by stopwords in both datasets, and `cosine_sim_19` gave 0.2004 for two unrelated posts that share only "the" and "in", which confirms what I could only suspect for `cosine_sim_8` in the pipeline: stopwords alone create similarity.

The Reddit data also showed two things the pipeline did not. The first is class imbalance: only 12.9% of the posts are positive, which caused the tied Pearson values driven by a handful of rare words. The second is the difference between topic and sentiment classification. 20-newsgroups categories are separated by topic, so they use clearly different words, like "graphics" versus "god". Positive, negative and neutral posts use heavily overlapping words and differ only in tone, context and financial slang, like "red in the ticker" labeled as neutral. This is why a bag-of-words model struggles with sentiment, and why t-SNE and UMAP failed completely here although they worked reasonably well on topics.

My tools generalized well: all 18 loaded and ran on a new dataset with three labels and much shorter texts without any errors, and `list_files_tool` found the file as in its known answer. One tool showed a limitation. `sample_data_tool` does not report the category counts of its sample; I had suggested adding them during AgentDev but we never did, so I skipped it and could not check the sampling risk from Master Q3 on this imbalanced data. The reduction tools themselves ran correctly; the failure to separate sentiment comes from the bag-of-words representation of very short texts, not from the algorithms.