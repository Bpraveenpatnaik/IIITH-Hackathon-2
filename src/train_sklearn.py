"""EDA and model comparison for house-price prediction with scikit-learn.

Run: python src/train_sklearn.py
Saves figures to figures/ and prints a results table.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV  # noqa: E402
from sklearn.metrics import mean_squared_error, r2_score  # noqa: E402
from sklearn.model_selection import KFold, cross_val_score, train_test_split  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

from data import load_data  # noqa: E402

FIGURES = Path(__file__).resolve().parent.parent / "figures"
TARGET = "SalePrice"
MISSING_THRESHOLD = 0.15  # drop columns missing in more than 15% of rows
SEED = 42


def eda(df: pd.DataFrame) -> list[str]:
    """Save the EDA figures and return the 10 features most correlated with price."""
    FIGURES.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(df[TARGET], ax=axes[0], color="#3a6fd8")
    axes[0].set_title("Sale price (skewed right)")
    sns.histplot(np.log1p(df[TARGET]), ax=axes[1], color="#3a6fd8")
    axes[1].set_title("log(1 + sale price), close to normal")
    fig.tight_layout()
    fig.savefig(FIGURES / "target_distribution.png", dpi=150)
    plt.close(fig)

    corr = df.select_dtypes("number").corr()
    top = corr.nlargest(10, TARGET)[TARGET].index.tolist()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(df[top].corr(), annot=True, fmt=".2f", cmap="Blues", square=True, ax=ax)
    ax.set_title("Top 10 features correlated with sale price")
    fig.tight_layout()
    fig.savefig(FIGURES / "top_correlations.png", dpi=150)
    plt.close(fig)

    missing = df.isnull().mean().sort_values(ascending=False)
    print("Columns with the most missing values:")
    print(missing[missing > 0].head(10).round(3).to_string(), "\n")
    return top[1:]


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes("number").columns
    categorical = X.columns.difference(numeric)
    return ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                              OneHotEncoder(handle_unknown="ignore")), categorical),
    ])


def main():
    df = load_data()
    print(f"Loaded {df.shape[0]} houses with {df.shape[1] - 1} features\n")
    eda(df)

    # drop columns that are mostly empty (PoolQC, MiscFeature, Alley, Fence, ...)
    missing = df.isnull().mean()
    dropped = missing[missing > MISSING_THRESHOLD].index.tolist()
    print(f"Dropping {len(dropped)} mostly-empty columns: {', '.join(dropped)}\n")
    X = df.drop(columns=dropped + [TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    pre = build_preprocessor(X)
    models = {
        "Linear regression": LinearRegression(),
        "Ridge (CV alpha)": RidgeCV(alphas=np.logspace(-2, 3, 30)),
        "Lasso (CV alpha)": LassoCV(cv=5, random_state=SEED, max_iter=50000),
        "PCA (5 components) + linear": make_pipeline(PCA(n_components=5), LinearRegression()),
        "Random forest": RandomForestRegressor(n_estimators=300, max_features=0.5,
                                               random_state=SEED, n_jobs=-1),
    }

    # PCA needs dense input, so one-hot output is made dense for every model
    pre.set_params(cat__onehotencoder__sparse_output=False)
    kfold = KFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    for name, model in models.items():
        # train on log(price): prices are skewed, and Kaggle scores RMSE on the log scale
        pipe = make_pipeline(pre, model)
        cv_rmse = -cross_val_score(pipe, X_train, np.log1p(y_train), cv=kfold,
                                   scoring="neg_root_mean_squared_error")
        pipe.fit(X_train, np.log1p(y_train))
        pred_log = pipe.predict(X_test)
        # clip before converting back to dollars so one wild prediction can't overflow
        pred = np.expm1(np.clip(pred_log, 0, np.log1p(y.max()) * 1.5))
        rows.append({
            "Model": name,
            "CV RMSE (log price)": np.median(cv_rmse),
            "Worst fold": cv_rmse.max(),
            "Test RMSE (log price)": np.sqrt(mean_squared_error(np.log1p(y_test), pred_log)),
            "Test RMSE ($)": np.sqrt(mean_squared_error(y_test, pred)),
            "Test R²": r2_score(y_test, pred),
        })

    results = pd.DataFrame(rows).sort_values("CV RMSE (log price)")
    print(results.to_markdown(index=False, floatfmt=(".0f", ".4f", ".4f", ".4f", ",.0f", ".3f")))


if __name__ == "__main__":
    main()
