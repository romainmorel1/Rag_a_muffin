import re
import numpy as np
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

OFFTOPIC = re.compile(r"\b(pizza|lasagnes|burger|tacos|sushi|steak)\b", re.IGNORECASE)

def refusal_message() -> str:
    return ("Je ne cuisine que des muffins 🧁😄\n"
            "Mais je peux te proposer un **muffin salé** dans ce style. Dis-moi tes ingrédients !")

def mmr(query_vec, doc_vecs, k=5, lambda_mult=0.65):
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

def build_context(metas, max_items=5) -> str:
    chunks = []
    for i, m in enumerate(metas[:max_items], start=1):
        title = m.get("title_fr") or m.get("title") or "Sans titre"
        ing = m.get("ingredients_fr") or ""
        # link = m.get("link") or ""
        # source = m.get("source") or ""
        # chunks.append(
        #     f"--- Recette {i} ---\n"
        #     f"Titre: {title}\n"
        #     f"Ingrédients: {ing}\n"
        #     f"Source: {source}\n"
        #     f"Lien: {link}\n"
        # )
        chunks.append(
            f"--- Recette {i} ---\n"
            f"Titre: {title}\n"
            f"Ingrédients: {ing}\n"
        )
    return "\n".join(chunks).strip()

class MuffinRAG:
    def __init__(self,
                 chroma_dir="data/chroma",
                 collection_name="royaume_du_muffin_e5",
                 embedding_model="intfloat/multilingual-e5-small"):
        self.client = chromadb.Client(ChromaSettings(is_persistent=True, persist_directory=chroma_dir))
        self.col = self.client.get_collection(collection_name)
        self.embed = SentenceTransformer(embedding_model)

    def retrieve(self, query: str, n_candidates=30, top_k=5, lambda_mmr=0.65):
        q_emb = self.embed.encode(["query: " + query], normalize_embeddings=True).tolist()
        res = self.col.query(
            query_embeddings=q_emb,
            n_results=n_candidates,
            include=["metadatas", "embeddings"]
        )
        metas = res["metadatas"][0]
        embs = np.array(res["embeddings"][0], dtype=np.float32)

        if not metas:
            return []

        chosen = mmr(np.array(q_emb[0], dtype=np.float32), embs, k=min(top_k, len(metas)), lambda_mult=lambda_mmr)
        return [metas[i] for i in chosen]

    def answer(self, query: str, llm, system_template: str):
        if OFFTOPIC.search(query):
            return refusal_message()

        metas = self.retrieve(query, top_k=5)

        if not metas:
            return ("Je suis prêt à te proposer un muffin… mais je ne trouve rien de pertinent dans mon grimoire 🧁\n"
                    "Ajoute quelques ingrédients (ex: farine, œufs, lait, fromage, chocolat…).")

        context_str = build_context(metas)
        system_prompt = system_template.format(context_str=context_str)
        return llm.generate(system_prompt=system_prompt, user_prompt=query, temperature=0.3)
