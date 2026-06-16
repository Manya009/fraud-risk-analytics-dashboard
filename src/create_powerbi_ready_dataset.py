import pandas as pd


SOURCE_FILE = "fraudTest_explainability_10000.csv"
OUTPUT_FILE = "PowerBI_Ready_FraudData.csv"


def build_powerbi_ready_dataset(source_file: str, output_file: str) -> None:
    df = pd.read_csv(source_file)

    df["revenue_at_risk"] = df["amt"] * df["predict_proba_fraud"]

    core_columns = [
        "Unnamed: 0",
        "trans_date_trans_time",
        "amt",
        "category",
        "job",
        "is_fraud",
        "predict_proba_fraud",
        "revenue_at_risk",
    ]

    records = []
    for rank in range(1, 4):
        temp_df = df[
            core_columns
            + [f"top{rank}_shap_feature", f"top{rank}_shap_value"]
        ].copy()

        temp_df = temp_df.rename(
            columns={
                f"top{rank}_shap_feature": "shap_feature_name",
                f"top{rank}_shap_value": "shap_value",
            }
        )
        temp_df["shap_rank"] = rank
        records.append(temp_df)

    final_df = pd.concat(records, ignore_index=True)

    final_df["shap_feature_name"] = (
        final_df["shap_feature_name"]
        .astype(str)
        .str.replace("num__", "", regex=False)
        .str.replace("cat__", "", regex=False)
    )

    final_df.to_csv(output_file, index=False)
    print(f"Saved Power BI-ready dataset to: {output_file}")
    print(f"Rows written: {len(final_df)}")


if __name__ == "__main__":
    build_powerbi_ready_dataset(SOURCE_FILE, OUTPUT_FILE)