"""
generate_data.py
-----------------
Simulates document embeddings for a small text-classification style
corpus, without needing a real embedding model.

In real NLP pipelines, you'd embed documents with something like
sentence-transformers or OpenAI/Anthropic embeddings (typically 384-1536
dimensional vectors). To keep this project self-contained and fast to run,
we instead simulate that high-dimensional embedding space directly using
`sklearn.datasets.make_blobs`, which is the standard textbook way to
generate clusterable synthetic data with controllable separation and noise.

Each "document" is labeled with a synthetic topic so we have ground-truth
to compare our unsupervised clustering against later.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

RANDOM_STATE = 42

TOPICS = [
    "ml_learning_booklets",
    "nlp_learning_booklets",
    "data_science_learning_booklets",
]

N_PER_TOPIC = [18, 14, 14]
EMBEDDING_DIM = 50  # stand-in for a real sentence-embedding dimensionality
CLUSTER_STD = 2.3


def generate_synthetic_embeddings(
    n_per_topic=N_PER_TOPIC,
    dim=EMBEDDING_DIM,
    cluster_std=CLUSTER_STD,
    random_state=RANDOM_STATE,
):
    """Generate high-dimensional synthetic 'document embeddings'.

    Returns
    -------
    X : np.ndarray, shape (n_samples, dim)
    labels : list[str], ground-truth topic per document
    """
    rng = np.random.RandomState(random_state)
    n_centers = len(n_per_topic)
    centers = rng.uniform(-6, 6, size=(n_centers, dim))

    X, y = make_blobs(
        n_samples=n_per_topic,
        n_features=dim,
        centers=centers,
        cluster_std=cluster_std,
        random_state=random_state,
    )
    labels = [TOPICS[i] for i in y]
    return X, labels


def build_dataframe(X, labels):
    doc_ids = [f"doc_{i:03d}" for i in range(len(labels))]
    df = pd.DataFrame(X, columns=[f"emb_{i}" for i in range(X.shape[1])])
    df.insert(0, "doc_id", doc_ids)
    df.insert(1, "true_topic", labels)
    return df


if __name__ == "__main__":
    X, labels = generate_synthetic_embeddings()
    df = build_dataframe(X, labels)
    df.to_csv("data/synthetic_embeddings.csv", index=False)
    print(f"Wrote {len(df)} synthetic embeddings to data/synthetic_embeddings.csv")
    print(df["true_topic"].value_counts())
