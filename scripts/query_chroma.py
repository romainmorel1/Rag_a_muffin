from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings as ChromaSettings

CHROMA_DIR = "data/chroma"
COLLECTION_NAME = "royaume_du_muffin"
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

import numpy as np

def mmr(query_vec, doc_vecs, k=5, lambda_mult=0.6):
    """
    Maximal Marginal Relevance
    query_vec: (d,)
    doc_vecs: (n,d)
    return: indices sélectionnés (k)
    """
    doc_vecs = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-12)
    query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-12)

    sims_to_query = doc_vecs @ query_vec
    selected = [int(np.argmax(sims_to_query))]
    candidates = set(range(len(doc_vecs))) - set(selected)

    for _ in range(k - 1):
        mmr_scores = []
        for c in candidates:
            sim_q = sims_to_query[c]
            sim_sel = max(doc_vecs[c] @ doc_vecs[s] for s in selected)
            score = lambda_mult * sim_q - (1 - lambda_mult) * sim_sel
            mmr_scores.append((score, c))
        _, best = max(mmr_scores, key=lambda x: x[0])
        selected.append(int(best))
        candidates.remove(best)

    return selected


def main():
    client = chromadb.Client(
        ChromaSettings(is_persistent=True, persist_directory=CHROMA_DIR)
    )
    collection = client.get_collection(COLLECTION_NAME)

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    queries = [
        "J'ai du fromage de chèvre et des épinards",
        "Je veux un muffin au chocolat bien moelleux",
        "Il me reste des bananes trop mûres",
        "Je veux une pizza quatre fromages"
    ]

    for q in queries:
        q_emb = model.encode([q]).tolist()
        res = collection.query(query_embeddings=q_emb, n_results=3)

        print("\n" + "="*80)
        print("🔎 Question:", q)

        metas = res["metadatas"][0]
        for i, m in enumerate(metas, start=1):
            print(f"  {i}. {m.get('title_fr','(sans titre)')}  | source={m.get('source','?')}")

if __name__ == "__main__":
    main()
