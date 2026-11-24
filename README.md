# House Price Prediction (IIIT Hyderabad Hackathon 2)

Predicting house sale prices in Ames, Iowa, from 79 features describing each house. This is the dataset from Kaggle's [House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) competition. The project goes through EDA and feature cleaning, compares scikit-learn models, and then rebuilds the pipeline in **PySpark MLlib** to show it can scale.

## Results

Hold-out test set: 20% of 1,460 houses. Models are trained on log(price), which is also the scale Kaggle uses for scoring.

| Model | CV RMSE (log price) | Test RMSE ($) | Test R² |
|---|---|---|---|
| **Lasso (alpha chosen by CV)** | **0.130** | **23,621** | **0.927** |
| Ridge (alpha chosen by CV) | 0.135 | 25,480 | 0.915 |
| Random forest (300 trees) | 0.144 | 29,120 | 0.889 |
| PCA (5 components) + linear regression | 0.148 | 31,467 | 0.871 |
| Plain linear regression | unstable | n/a | n/a |
| PySpark elastic-net regression (36 numeric features) | 0.147 (test) | 33,198 | 0.887 (log scale) |

**Findings**
- **Regularization is essential here.** After one-hot encoding there are about 270 correlated columns. Plain linear regression produces huge predictions on some cross-validation folds, while Lasso and Ridge stay stable and do best.
- **The log transform helps.** Sale prices are skewed to the right, and taking the log makes them close to normal (see `figures/target_distribution.png`).
- **The strongest single feature is overall quality** (correlation 0.79 with price), followed by above-ground living area (0.71) and garage size (see `figures/top_correlations.png`).
- **Five PCA components keep most of the signal** but cost about $8k in accuracy compared with Lasso.
- **Spark without categorical features** comes close to the scikit-learn models, so most of the signal is in the numeric features.

## Approach
1. **EDA:** the distribution of the target, a correlation heatmap of the top 10 features, and a table of missing values.
2. **Cleaning:** drop the 7 columns missing in more than 15% of rows (e.g. `PoolQC`, `Alley`, `Fence`, `FireplaceQu`). Fill other gaps with the median for numbers and the most frequent value for categories, then one-hot encode and scale.
3. **Modelling:** scikit-learn `Pipeline` + `ColumnTransformer`, so the preprocessing is learned only from the training folds (no data leakage), with 5-fold cross-validation.
4. **Spark:** `Imputer` → `VectorAssembler` → `StandardScaler` → elastic-net `LinearRegression` in a Spark ML `Pipeline`. The target is kept out of the feature vector.

## Run it

```bash
pip install -r requirements.txt
cd src
python train_sklearn.py   # EDA figures + model comparison
python train_spark.py     # PySpark version (needs Java 8+)
```

The scripts use `data/train.csv` if it exists. Otherwise they download the same data from OpenML.

The original Colab notebook from the hackathon is kept in `archive/` for reference.
