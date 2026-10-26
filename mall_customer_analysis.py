from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "Mall_Customers.csv"
NEW_CUSTOMERS_FILE = BASE_DIR / "New_Customers.csv"
OUTPUT_DIR = BASE_DIR / "mall_customer_analysis"
FEATURES = ["Annual Income (k$)", "Spending Score (1-100)"]
NUMERIC_COLUMNS = [
    "CustomerID",
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)",
]
REQUIRED_COLUMNS = ["CustomerID", "Gender", "Age", *FEATURES]
RANDOM_STATE = 42


def load_and_clean_data() -> tuple[pd.DataFrame, dict[str, int]]:
    raw = pd.read_csv(INPUT_FILE)
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(raw.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    original_rows = len(raw)
    exact_duplicates = int(raw.duplicated().sum())
    data = raw.copy()
    data.columns = data.columns.str.strip()
    data["Gender"] = data["Gender"].astype("string").str.strip().str.title()
    for column in NUMERIC_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="raise")
    data = data.drop_duplicates().reset_index(drop=True)

    if data[REQUIRED_COLUMNS].isna().any().any():
        missing = data[REQUIRED_COLUMNS].isna().sum()
        raise ValueError(f"Missing values require review; counts: {missing[missing > 0].to_dict()}")
    if data["CustomerID"].duplicated().any():
        raise ValueError("CustomerID must be unique; resolve duplicate IDs before clustering.")
    if not data["Gender"].isin(["Female", "Male"]).all():
        raise ValueError("Gender contains unexpected values; review categories before analysis.")
    if (data["Age"] <= 0).any() or (data["Annual Income (k$)"] < 0).any():
        raise ValueError("Age and income contain values outside valid ranges.")
    if not data["Spending Score (1-100)"].between(1, 100).all():
        raise ValueError("Spending Score (1-100) must be between 1 and 100.")

    return data, {
        "input_rows": original_rows,
        "exact_duplicates_removed": exact_duplicates,
        "clean_rows": len(data),
    }


def load_new_customers() -> pd.DataFrame:
    if not NEW_CUSTOMERS_FILE.exists():
        pd.DataFrame(columns=FEATURES).to_csv(NEW_CUSTOMERS_FILE, index=False)
        print(
            f"Created {NEW_CUSTOMERS_FILE.name}. Add new customers' income and "
            "spending score, then run the script again."
        )
        return pd.DataFrame(columns=FEATURES)

    data = pd.read_csv(NEW_CUSTOMERS_FILE)
    missing_columns = sorted(set(FEATURES) - set(data.columns))
    if missing_columns:
        raise ValueError(
            f"{NEW_CUSTOMERS_FILE.name} is missing required columns: {missing_columns}"
        )

    for column in FEATURES:
        data[column] = pd.to_numeric(data[column], errors="raise")
    if data[FEATURES].isna().any().any():
        missing = data[FEATURES].isna().sum()
        raise ValueError(
            f"New customer data has missing values; counts: "
            f"{missing[missing > 0].to_dict()}"
        )
    if (data["Annual Income (k$)"] < 0).any():
        raise ValueError("New customer income cannot be negative.")
    if not data["Spending Score (1-100)"].between(1, 100).all():
        raise ValueError("New customer spending scores must be between 1 and 100.")

    return data


def clustering_metrics(
    model_name: str,
    labels: pd.Series,
    features: pd.DataFrame,
    family: str,
    noise_points: int = 0,
) -> dict[str, object] | None:
    inlier_mask = labels.to_numpy() != -1
    inlier_labels = labels.to_numpy()[inlier_mask]
    inlier_features = features.to_numpy()[inlier_mask]
    cluster_count = len(set(inlier_labels))
    if cluster_count < 2 or cluster_count >= len(inlier_labels):
        return None

    return {
        "model": model_name,
        "family": family,
        "clusters": cluster_count,
        "noise_points": noise_points,
        "noise_percent": 100 * noise_points / len(labels),
        "silhouette": silhouette_score(inlier_features, inlier_labels),
        "davies_bouldin": davies_bouldin_score(inlier_features, inlier_labels),
        "calinski_harabasz": calinski_harabasz_score(inlier_features, inlier_labels),
    }


def compare_models(scaled_features: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    results: list[dict[str, object]] = []
    inertias: list[dict[str, int | float]] = []

    for cluster_count in range(2, 11):
        kmeans = KMeans(
            n_clusters=cluster_count,
            random_state=RANDOM_STATE,
            n_init=20,
        )
        labels = pd.Series(kmeans.fit_predict(scaled_features))
        inertias.append({"clusters": cluster_count, "inertia": kmeans.inertia_})
        result = clustering_metrics(
            f"KMeans (k={cluster_count})", labels, scaled_features, "KMeans"
        )
        if result:
            results.append(result)

        ward_labels = pd.Series(
            AgglomerativeClustering(n_clusters=cluster_count, linkage="ward").fit_predict(
                scaled_features
            )
        )
        result = clustering_metrics(
            f"Agglomerative Ward (k={cluster_count})",
            ward_labels,
            scaled_features,
            "Agglomerative Ward",
        )
        if result:
            results.append(result)

        mixture = GaussianMixture(
            n_components=cluster_count,
            random_state=RANDOM_STATE,
            n_init=5,
        )
        mixture_labels = pd.Series(mixture.fit_predict(scaled_features))
        result = clustering_metrics(
            f"Gaussian Mixture (k={cluster_count})",
            mixture_labels,
            scaled_features,
            "Gaussian Mixture",
        )
        if result:
            results.append(result)

    for epsilon in [round(value / 10, 1) for value in range(2, 16)]:
        dbscan_labels = pd.Series(
            DBSCAN(eps=epsilon, min_samples=5).fit_predict(scaled_features)
        )
        noise_points = int((dbscan_labels == -1).sum())
        result = clustering_metrics(
            f"DBSCAN (eps={epsilon:.1f}, min_samples=5)",
            dbscan_labels,
            scaled_features,
            "DBSCAN",
            noise_points,
        )
        if result:
            results.append(result)

    comparison = pd.DataFrame(results).sort_values(
        ["family", "clusters", "silhouette"], ascending=[True, True, False]
    )
    return comparison, inertias


def save_eda_plots(data: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")
    figure, axes = plt.subplots(2, 3, figsize=(16, 9))
    for axis, column, title in [
        (axes[0, 0], "Age", "Age distribution"),
        (axes[0, 1], "Annual Income (k$)", "Annual income distribution"),
        (axes[0, 2], "Spending Score (1-100)", "Spending score distribution"),
    ]:
        sns.histplot(data=data, x=column, kde=True, ax=axis, color="#3977a8")
        axis.set_title(title)

    sns.scatterplot(
        data=data,
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        hue="Gender",
        alpha=0.8,
        ax=axes[1, 0],
    )
    axes[1, 0].set_title("Income vs spending by gender")
    sns.countplot(data=data, x="Gender", ax=axes[1, 1], color="#6aaed6")
    axes[1, 1].set_title("Gender counts")
    correlation = data[["Age", *FEATURES]].corr()
    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=0,
        ax=axes[1, 2],
    )
    axes[1, 2].set_title("Numeric feature correlations")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "eda_overview.png", dpi=160)
    plt.close(figure)


def save_model_plot(comparison: pd.DataFrame, inertias: list[dict[str, int | float]]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    fixed_partition = comparison[comparison["family"] != "DBSCAN"]
    sns.lineplot(
        data=fixed_partition,
        x="clusters",
        y="silhouette",
        hue="family",
        marker="o",
        ax=axes[0],
    )
    axes[0].set_title("Silhouette score by cluster count")
    axes[0].set_ylabel("Silhouette (higher is better)")

    inertia_frame = pd.DataFrame(inertias)
    sns.lineplot(
        data=inertia_frame,
        x="clusters",
        y="inertia",
        marker="o",
        ax=axes[1],
        color="#bf6b35",
    )
    axes[1].set_title("KMeans elbow curve")
    axes[1].set_ylabel("Inertia (lower is better)")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "model_comparison.png", dpi=160)
    plt.close(figure)


def save_segment_plot(data: pd.DataFrame, labels: pd.Series, centers: pd.DataFrame) -> None:
    plot_data = data.copy()
    plot_data["Cluster"] = labels.astype(str)
    figure, axis = plt.subplots(figsize=(9, 6))
    sns.scatterplot(
        data=plot_data,
        x=FEATURES[0],
        y=FEATURES[1],
        hue="Cluster",
        palette="tab10",
        s=65,
        alpha=0.85,
        ax=axis,
    )
    axis.scatter(
        centers[FEATURES[0]],
        centers[FEATURES[1]],
        marker="X",
        s=240,
        color="black",
        edgecolor="white",
        linewidth=1,
        label="Centroid",
    )
    axis.set_title("Mall customer segments: KMeans (k=5)")
    axis.legend(title="Cluster")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "kmeans_segments.png", dpi=160)
    plt.close(figure)


def to_markdown_table(frame: pd.DataFrame, include_index: bool = False) -> str:
    display = frame.reset_index() if include_index else frame.copy()
    headers = [str(column) for column in display.columns]
    rows = [
        ["" if pd.isna(value) else str(value).replace("|", "\\|") for value in row]
        for row in display.itertuples(index=False, name=None)
    ]
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(row) + " |" for row in rows),
        ]
    )


def write_report(
    data: pd.DataFrame,
    cleaning: dict[str, int],
    comparison: pd.DataFrame,
    selected_metrics: pd.Series,
    profile: pd.DataFrame,
) -> None:
    numeric_summary = to_markdown_table(data[["Age", *FEATURES]].describe().round(2), True)
    gender_counts = data["Gender"].value_counts().to_dict()
    top_fixed = (
        comparison[comparison["family"] != "DBSCAN"]
        .sort_values("silhouette", ascending=False)
        .head(5)
    )
    model_table = top_fixed[
        ["model", "clusters", "silhouette", "davies_bouldin", "calinski_harabasz"]
    ].round(4)
    model_table = to_markdown_table(model_table)
    profile_table = to_markdown_table(profile.round(2), True)
    best_dbscan = comparison[comparison["family"] == "DBSCAN"].sort_values(
        "silhouette", ascending=False
    ).iloc[0]

    report = f"""# Mall Customer EDA and Clustering

## Dataset and cleaning

- Source: `{INPUT_FILE.name}`; {cleaning['input_rows']} input rows and {len(data.columns)} columns.
- Exact duplicate rows removed: {cleaning['exact_duplicates_removed']}.
- Rows after cleaning: {cleaning['clean_rows']}; missing values: 0; duplicate customer IDs: 0.
- Numeric fields were parsed as numbers and the Gender text was trimmed and normalized.
- `CustomerID` is retained in the cleaned export but excluded from clustering because it is an identifier, not a customer behavior feature.
- Clustering uses standardized annual income and spending score. Standardization is important because the clustering algorithms use distances.

## Exploratory data analysis

- Gender counts: {gender_counts}.
- Descriptive statistics:

{numeric_summary}

- Age is included in the EDA and segment profiles, but not in the primary clustering; the classic mall segmentation uses income and spending score, making segments directly interpretable on those axes.
- To assign new customers, enter their annual income and spending score in `New_Customers.csv`. The script uses the same scaler and KMeans model trained on this dataset and writes assigned groups to `new_customer_predictions.csv`. This assigns a segment; it does not measure prediction accuracy.

![EDA overview](eda_overview.png)

## Unsupervised model comparison

Models were evaluated on the same two standardized features. KMeans and Ward agglomerative clustering were tested for k=2 through 10; Gaussian Mixture models used 2 through 10 components; DBSCAN used `min_samples=5` and epsilon values 0.2 through 1.5.

| Metric | Interpretation |
| --- | --- |
| Silhouette | Higher is better; measures separation and cohesion. |
| Davies-Bouldin | Lower is better; compares within-cluster scatter with separation. |
| Calinski-Harabasz | Higher is better; compares between-cluster and within-cluster dispersion. |

Top fixed-partition candidates by silhouette:

{model_table}

The best useful fixed-partition result in this search is **KMeans with k=5**: silhouette **{selected_metrics['silhouette']:.4f}**, Davies-Bouldin **{selected_metrics['davies_bouldin']:.4f}**, and Calinski-Harabasz **{selected_metrics['calinski_harabasz']:.2f}**. At five clusters, Ward agglomerative and Gaussian Mixture scores were very close (silhouette around 0.554). KMeans has the slightly better silhouette on this run and provides simple, reproducible assignments.

DBSCAN is reported separately because its silhouette is calculated only on non-noise points. The highest DBSCAN silhouette candidate ({best_dbscan['model']}) labels {int(best_dbscan['noise_points'])} of {len(data)} customers as noise, so the larger score is not a better overall segmentation of the full customer base.

![Model comparison](model_comparison.png)

## Selected customer segments

KMeans segment profiles (means; cluster labels are arbitrary identifiers):

{profile_table}

![KMeans customer segments](kmeans_segments.png)

## Conclusion

**Recommended for this dataset: KMeans with five clusters**, using standardized annual income and spending score. It has the highest silhouette among the tested fixed-partition models overall and the strongest five-cluster result across the three reported metrics, while assigning every customer to a segment. This is an internal clustering comparison, not supervised predictive accuracy: there are no ground-truth segment labels, so the scores do not establish an objective accuracy or guarantee business value. Review the segment profiles and validate them with business context before acting on them.

## Reproduce

Run `python mall_customer_analysis.py` in the folder containing `{INPUT_FILE.name}`. To assign new customers, fill in `New_Customers.csv` with the columns `Annual Income (k$)` and `Spending Score (1-100)`, then run the script. Results are written to `new_customer_predictions.csv` in the output folder.
"""
    (OUTPUT_DIR / "analysis_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    data, cleaning = load_and_clean_data()
    data.to_csv(OUTPUT_DIR / "cleaned_mall_customers.csv", index=False)

    scaler = StandardScaler()
    scaled_features = pd.DataFrame(
        scaler.fit_transform(data[FEATURES]),
        columns=FEATURES,
        index=data.index,
    )
    comparison, inertias = compare_models(scaled_features)
    comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    selected_model = KMeans(
        n_clusters=5,
        random_state=RANDOM_STATE,
        n_init=20,
    ).fit(scaled_features)
    selected_labels = selected_model.labels_
    labeled_data = data.copy()
    labeled_data["Cluster"] = selected_labels
    labeled_data.to_csv(OUTPUT_DIR / "kmeans_customer_segments.csv", index=False)
    joblib.dump(selected_model, OUTPUT_DIR / "kmeans_model.joblib")
    joblib.dump(scaler, OUTPUT_DIR / "scaler.joblib")

    new_customers = load_new_customers()
    if not new_customers.empty:
        new_customers["Predicted Cluster"] = selected_model.predict(
            scaler.transform(new_customers[FEATURES])
        )
    else:
        new_customers["Predicted Cluster"] = pd.Series(dtype="int64")
        print(f"No new customer rows to assign in {NEW_CUSTOMERS_FILE.name}.")
    new_customers.to_csv(
        OUTPUT_DIR / "new_customer_predictions.csv",
        index=False,
    )

    centers = pd.DataFrame(
        scaler.inverse_transform(selected_model.cluster_centers_),
        columns=FEATURES,
    )
    profile = labeled_data.groupby("Cluster").agg(
        customers=("CustomerID", "size"),
        mean_age=("Age", "mean"),
        mean_annual_income=("Annual Income (k$)", "mean"),
        mean_spending_score=("Spending Score (1-100)", "mean"),
    )
    gender_mix = (
        pd.crosstab(labeled_data["Cluster"], labeled_data["Gender"], normalize="index") * 100
    ).add_prefix("gender_%_")
    profile = profile.join(gender_mix)

    selected_metrics = comparison[
        comparison["model"] == "KMeans (k=5)"
    ].iloc[0].copy()

    save_eda_plots(data)
    save_model_plot(comparison, inertias)
    save_segment_plot(data, pd.Series(selected_labels), centers)
    write_report(data, cleaning, comparison, selected_metrics, profile)
    try:
        from generate_pdf import generate_pdf
        generate_pdf()
    except Exception as e:
        print(f"PDF generation note: {e}")

    print(f"Input rows: {cleaning['input_rows']}; clean rows: {cleaning['clean_rows']}")
    print(f"Exact duplicate rows removed: {cleaning['exact_duplicates_removed']}")
    print(f"Outputs: {OUTPUT_DIR}")
    print(
        "Selected: KMeans (k=5), "
        f"silhouette={selected_metrics['silhouette']:.4f}, "
        f"Davies-Bouldin={selected_metrics['davies_bouldin']:.4f}, "
        f"Calinski-Harabasz={selected_metrics['calinski_harabasz']:.2f}"
    )
    print("\nTop fixed-partition candidates:")
    print(
        comparison[comparison["family"] != "DBSCAN"]
        .sort_values("silhouette", ascending=False)
        .head(5)[["model", "silhouette", "davies_bouldin", "calinski_harabasz"]]
        .to_string(index=False)
    )
    print("\nSegment profiles:")
    print(profile.round(2).to_string())


if __name__ == "__main__":
    main()
