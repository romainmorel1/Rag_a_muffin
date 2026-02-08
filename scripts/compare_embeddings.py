import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap
from sentence_transformers import SentenceTransformer

PARQUET_PATH = "data/processed/muffins_fr.parquet"
N_SAMPLE = 2000  # ajuste selon ton PC

MODELS = [
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "intfloat/multilingual-e5-small",
]

QUERIES = [
    "Je veux un muffin au chocolat bien moelleux",
    "Il me reste des bananes trop mûres",
    "J'ai du fromage et des épinards",
    "Je veux un muffin citron pavot",
]

def label_sucre_sale(text: str) -> str:
    t = (text or "").lower()
    sucre = any(w in t for w in ["sucre", "chocolat", "vanille", "miel", "confiture", "banane", "myrtille", "citron"])
    sale = any(w in t for w in ["fromage", "chèvre", "jambon", "lardon", "thon", "feta", "épinard", "tomate"])
    if sucre and not sale:
        return "sucré"
    if sale and not sucre:
        return "salé"
    if sucre and sale:
        return "mixte"
    return "inconnu"

def build_docs(df: pd.DataFrame) -> list[str]:
    return (df["title_fr"].fillna("") + "\nIngrédients:\n" + df["ingredients_fr"].fillna("")).tolist()

def encode_docs(model_name: str, docs: list[str]) -> np.ndarray:
    model = SentenceTransformer(model_name)
    # E5 marche souvent mieux avec préfixes query/passage
    if "e5" in model_name:
        docs = ["passage: " + d for d in docs]
    emb = model.encode(docs, show_progress_bar=True, normalize_embeddings=True)
    return emb, model

def encode_query(model_name: str, model, q: str) -> np.ndarray:
    if "e5" in model_name:
        q = "query: " + q
    return model.encode([q], normalize_embeddings=True)

def topk_sim(emb_docs: np.ndarray, emb_q: np.ndarray, k: int = 5):
    sims = emb_docs @ emb_q.T
    sims = sims.reshape(-1)
    idx = np.argsort(-sims)[:k]
    return idx, sims[idx]

def main():
    df = pd.read_parquet(PARQUET_PATH)
    df = df.sample(n=min(N_SAMPLE, len(df)), random_state=42).reset_index(drop=True)
    df["label"] = df["ingredients_fr"].apply(label_sucre_sale)

    docs = build_docs(df)

    for model_name in MODELS:
        print("\n" + "="*90)
        print("MODEL:", model_name)

        emb_docs, model = encode_docs(model_name, docs)
        print("Embedding shape:", emb_docs.shape, "=> dim:", emb_docs.shape[1])

        # --- UMAP ---
        reducer = umap.UMAP(n_neighbors=20, min_dist=0.1, random_state=42)
        xy = reducer.fit_transform(emb_docs)

        plt.figure()
        for lab in ["sucré", "salé", "mixte", "inconnu"]:
            mask = (df["label"] == lab).values
            plt.scatter(xy[mask, 0], xy[mask, 1], s=8, label=lab)
        plt.title(f"UMAP - {model_name}")
        plt.legend()
        plt.show()

        # --- Mini retrieval ---
        for q in QUERIES:
            emb_q = encode_query(model_name, model, q)
            idx, sims = topk_sim(emb_docs, emb_q, k=5)
            print("\n🔎", q)
            for rank, (i, s) in enumerate(zip(idx, sims), start=1):
                print(f"  {rank}. {df.loc[i, 'title_fr']}  (sim={float(s):.3f})")

if __name__ == "__main__":
    main()
