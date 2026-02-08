import numpy as np
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer


CHROMA_DIR = "data/chroma"
COLLECTION_NAME = "royaume_du_muffin_e5"
MODEL_NAME = "intfloat/multilingual-e5-small"

N_CANDIDATES = 30   # on récupère large...
TOP_K_FINAL = 5     # ...puis on diversifie
LAMBDA = 0.65       # 0.5 = très divers, 0.8 = très pertinent


def mmr(query_vec, doc_vecs, k=5, lambda_mult=0.65):
    """
    Maximal Marginal Relevance
    Retourne des indices diversifiés parmi doc_vecs.
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

    return selected, sims_to_query


def main():
    client = chromadb.Client(ChromaSettings(is_persistent=True, persist_directory=CHROMA_DIR))
    col = client.get_collection(COLLECTION_NAME)

    model = SentenceTransformer(MODEL_NAME)

    queries = [
        "J'ai du fromage de chèvre et des épinards",
        "Je veux un muffin au chocolat bien moelleux",
        "Il me reste des bananes trop mûres",
        "Je veux un muffin citron pavot",
        "Je veux une pizza quatre fromages",
    ]

    for q in queries:
        q_emb = model.encode(["query: " + q], normalize_embeddings=True).tolist()

        # On demande plus large pour diversifier ensuite
        res = col.query(
            query_embeddings=q_emb,
            n_results=N_CANDIDATES,
            include=["metadatas", "documents", "embeddings"]
        )

        metas = res["metadatas"][0]
        embs = np.array(res["embeddings"][0], dtype=np.float32)

        # MMR
        selected, sims_to_query = mmr(
            query_vec=np.array(q_emb[0], dtype=np.float32),
            doc_vecs=embs,
            k=min(TOP_K_FINAL, len(metas)),
            lambda_mult=LAMBDA
        )

        print("\n" + "=" * 90)
        print("🔎 Question:", q)
        for rank, i in enumerate(selected, start=1):
            title = metas[i].get("title_fr", "(sans titre)")
            src = metas[i].get("source", "?")
            sim = float(sims_to_query[i])
            print(f"  {rank}. {title}  | source={src} | sim={sim:.3f}")


if __name__ == "__main__":
    main()
