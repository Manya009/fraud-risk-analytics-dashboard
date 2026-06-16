import argparse
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data.columns = data.columns.str.strip()

    data["trans_date_trans_time"] = pd.to_datetime(data["trans_date_trans_time"], errors="coerce")
    data["dob"] = pd.to_datetime(data["dob"], errors="coerce")

    data["transaction_hour"] = data["trans_date_trans_time"].dt.hour
    data["transaction_day"] = data["trans_date_trans_time"].dt.day
    data["transaction_month"] = data["trans_date_trans_time"].dt.month
    data["transaction_dayofweek"] = data["trans_date_trans_time"].dt.dayofweek

    age = data["trans_date_trans_time"].dt.year - data["dob"].dt.year
    birthday_not_passed = (
        (data["trans_date_trans_time"].dt.month < data["dob"].dt.month)
        | (
            (data["trans_date_trans_time"].dt.month == data["dob"].dt.month)
            & (data["trans_date_trans_time"].dt.day < data["dob"].dt.day)
        )
    )
    data["customer_age"] = age - birthday_not_passed.fillna(False).astype(int)

    drop_columns = [
        "Unnamed: 0",
        "trans_date_trans_time",
        "dob",
        "trans_num",
        "first",
        "last",
        "street",
        "city",
        "zip",
        "lat",
        "long",
        "merch_lat",
        "merch_long",
        "cc_num",
        "merchant",
        "job",
    ]
    data = data.drop(columns=[column for column in drop_columns if column in data.columns])
    return data


def build_pipeline(X_train: pd.DataFrame) -> Pipeline:
    categorical_features = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_features = X_train.select_dtypes(exclude=["object", "category"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                        ),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    model = RandomForestClassifier(
        n_estimators=80,
        max_depth=18,
        max_samples=0.2,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced_subsample",
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def positive_class_shap(shap_values) -> np.ndarray:
    if isinstance(shap_values, list):
        return np.asarray(shap_values[1])

    values = np.asarray(shap_values)
    if values.ndim == 3:
        return values[:, :, 1]
    if values.ndim == 2:
        return values
    raise ValueError(f"Unexpected SHAP output shape: {values.shape}")


def top_three_features(
    shap_matrix: np.ndarray,
    feature_names: np.ndarray,
) -> Tuple[List[str], List[float], List[str], List[float], List[str], List[float]]:
    abs_matrix = np.abs(shap_matrix)
    top_idx = np.argpartition(abs_matrix, -3, axis=1)[:, -3:]

    row_order = np.arange(shap_matrix.shape[0])[:, None]
    top_scores = abs_matrix[row_order, top_idx]
    sort_order = np.argsort(top_scores, axis=1)[:, ::-1]
    ordered_idx = top_idx[row_order, sort_order]

    first_idx = ordered_idx[:, 0]
    second_idx = ordered_idx[:, 1]
    third_idx = ordered_idx[:, 2]

    top1_feature = feature_names[first_idx].tolist()
    top2_feature = feature_names[second_idx].tolist()
    top3_feature = feature_names[third_idx].tolist()

    top1_value = shap_matrix[np.arange(shap_matrix.shape[0]), first_idx].astype(float).tolist()
    top2_value = shap_matrix[np.arange(shap_matrix.shape[0]), second_idx].astype(float).tolist()
    top3_value = shap_matrix[np.arange(shap_matrix.shape[0]), third_idx].astype(float).tolist()

    return top1_feature, top1_value, top2_feature, top2_value, top3_feature, top3_value


def generate_explainability_dataset(
    train_path: str,
    test_path: str,
    output_path: str,
    chunk_size: int,
    max_test_rows: Optional[int],
    random_state: int,
) -> None:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if max_test_rows is not None and max_test_rows < len(test_df):
        test_df = test_df.sample(n=max_test_rows, random_state=random_state).copy()

    train_clean = engineer_features(train_df)
    test_clean = engineer_features(test_df)

    X_train = train_clean.drop(columns="is_fraud")
    y_train = train_clean["is_fraud"]
    X_test = test_clean.drop(columns="is_fraud")

    pipeline = build_pipeline(X_train)
    pipeline.fit(X_train, y_train)

    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    model: RandomForestClassifier = pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.TreeExplainer(model)

    enriched = test_df.copy()
    enriched["predict_proba_fraud"] = 0.0
    enriched["top1_shap_feature"] = ""
    enriched["top1_shap_value"] = 0.0
    enriched["top2_shap_feature"] = ""
    enriched["top2_shap_value"] = 0.0
    enriched["top3_shap_feature"] = ""
    enriched["top3_shap_value"] = 0.0

    total_rows = len(X_test)
    for start in range(0, total_rows, chunk_size):
        end = min(start + chunk_size, total_rows)
        X_chunk = X_test.iloc[start:end]

        X_chunk_t = preprocessor.transform(X_chunk)
        chunk_proba = model.predict_proba(X_chunk_t)[:, 1]
        chunk_shap = positive_class_shap(explainer.shap_values(X_chunk_t))

        t1f, t1v, t2f, t2v, t3f, t3v = top_three_features(chunk_shap, feature_names)

        enriched.loc[enriched.index[start:end], "predict_proba_fraud"] = chunk_proba
        enriched.loc[enriched.index[start:end], "top1_shap_feature"] = t1f
        enriched.loc[enriched.index[start:end], "top1_shap_value"] = t1v
        enriched.loc[enriched.index[start:end], "top2_shap_feature"] = t2f
        enriched.loc[enriched.index[start:end], "top2_shap_value"] = t2v
        enriched.loc[enriched.index[start:end], "top3_shap_feature"] = t3f
        enriched.loc[enriched.index[start:end], "top3_shap_value"] = t3v

        print(f"Processed {end}/{total_rows} rows")

    enriched.to_csv(output_path, index=False)
    print(f"Explainability dataset saved to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate test-set explainability CSV with predict_proba and top-3 SHAP contributors."
    )
    parser.add_argument("--train", default="fraudTrain.csv", help="Path to training CSV")
    parser.add_argument("--test", default="fraudTest.csv", help="Path to test CSV")
    parser.add_argument(
        "--output",
        default="fraudTest_explainability.csv",
        help="Output path for enriched explainability CSV",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=20000,
        help="Rows per chunk for SHAP computation",
    )
    parser.add_argument(
        "--max-test-rows",
        type=int,
        default=None,
        help="Optional cap for test rows, useful for smoke tests",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used when sampling max_test_rows from the test set",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_explainability_dataset(
        train_path=args.train,
        test_path=args.test,
        output_path=args.output,
        chunk_size=args.chunk_size,
        max_test_rows=args.max_test_rows,
        random_state=args.random_state,
    )