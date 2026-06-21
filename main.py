"""
main.py
-------
Runs the full micro-pipeline end to end:
  synthetic data generation -> PCA -> K-Means -> dark cube-marker plot
"""

import os
import pandas as pd

from src.generate_data import generate_synthetic_embeddings, build_dataframe
from src.cluster import reduce_to_3d, run_kmeans, summarize
from src.visualize import plot_cube_clusters

os.makedirs("data", exist_ok=True)
os.makedirs("output", exist_ok=True)


def main():
    X, labels = generate_synthetic_embeddings()
    df = build_dataframe(X, labels)
    df.to_csv("data/synthetic_embeddings.csv", index=False)

    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    X_3d, explained = reduce_to_3d(df[emb_cols].values)
    print(f"PCA explained variance ratio (3 components): {explained.round(3)} "
          f"(total: {explained.sum():.1%})")

    n_clusters = df["true_topic"].nunique()
    cluster_ids, score = run_kmeans(X_3d, n_clusters)
    print(f"K-Means silhouette score: {score:.3f}")

    df_out, crosstab = summarize(df, cluster_ids)
    print("\nCluster vs. ground-truth topic:")
    print(crosstab)

    for i, axis in enumerate(["pc1", "pc2", "pc3"]):
        df_out[axis] = X_3d[:, i]
    df_out.to_csv("data/clustered_embeddings.csv", index=False)

    plot_cube_clusters(df_out)


if __name__ == "__main__":
    main()
