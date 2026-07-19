# MSCS_634_Lab_2 — Classification Using KNN and RNN Algorithms

## Purpose

This lab explores the performance of two distance-based classification algorithms — **K-Nearest Neighbors (KNN)** and **Radius Neighbors (RNN)** — on the Wine dataset from `sklearn`. The dataset contains 178 samples across 3 wine classes, each described by 13 chemical properties (alcohol content, malic acid, ash, flavanoids, proline, etc.). The goal is to:

- Explore and prepare the dataset for classification (80/20 train/test split, feature scaling)
- Train and evaluate KNN classifiers across k = 1, 5, 11, 15, and 21
- Train and evaluate RNN classifiers across radius = 350, 400, 450, 500, 550, and 600
- Visualize accuracy trends for both models
- Compare the two approaches and discuss when each is preferable

## Key Insights

**KNN performed very well across all tested k values**, with accuracy ranging from 97.2% (k=1, k=5) to a perfect 100% (k=11, 15, 21). Once features were standardized, KNN's decision boundaries were stable and not overly sensitive to the choice of k, though very small k (k=1) was the most sensitive to noise since it relies on a single neighbor's vote.

**RNN, in contrast, produced a flat, low accuracy of 38.9% across every tested radius value.** Investigation showed this was exactly equal to the size of the largest class in the test set — meaning the classifier was effectively always predicting the majority class rather than genuinely using local neighbor information. The root cause: after standardizing the features, distances between points in the 13-dimensional space are small (typically single digits), so radius values in the hundreds (as specified by the lab) were far larger than the actual spread of the data. Every training point ended up within range of every test point, collapsing RNN into a majority-class predictor.

This comparison highlights an important practical lesson: **KNN's `k` parameter is scale-independent and easy to reason about, while RNN's `radius` parameter is highly sensitive to feature scaling** and must be chosen relative to the actual distances present in the (scaled) feature space. For standardized data, meaningful radius values for this dataset would likely be in the range of roughly 1–10 rather than in the hundreds.

**When to prefer each model:**
- **KNN** is generally the safer default — it adapts naturally to varying data density and only requires tuning a single integer. It clearly outperformed RNN in this experiment.
- **RNN** can still be valuable in applications where the *density* of neighbors within a meaningful distance matters more than a fixed count, or where you want the model to abstain from predicting in sparse regions of feature space. However, it requires careful, scale-aware tuning of the radius to be effective.

## Challenges and Decisions

- **Feature scaling**: KNN and RNN are both distance-based, so features were standardized using `StandardScaler` before training. This was essential for fair distance comparisons across features with very different natural ranges (e.g., `proline` values in the hundreds vs. `hue` values less than 2).
- **RNN outlier handling**: With smaller radius values (or a poorly scaled radius), some test points could have zero neighbors within range, which raises an error by default. This was handled by setting `outlier_label='most_frequent'` in `RadiusNeighborsClassifier`, so such points are assigned the majority class instead of crashing the model.
- **Interpreting the flat RNN accuracy curve**: The RNN accuracy trend was flat across all six radius values, which was not what the lab instructions implied a "trend" would look like. Rather than treating this as an error, the notebook investigates *why* this happened (radius values too large relative to standardized feature distances) and explains it as a meaningful finding about how RNN's radius parameter interacts with feature scaling.

## Files

- `MSCS_634_Lab_2.ipynb` — Jupyter Notebook containing all code, outputs, and analysis for the lab.
- `README.md` — This file.
