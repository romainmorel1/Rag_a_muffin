import uuid
from pathlib import Path

import pandas as pd
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# =========================
# CONFIG
# =========================
PROCESSED_PATH = Path("data/processed/muffins_fr.parquet")
CHROMA_DIR = Path("data/chroma")
COLLECTION_NAME = "royaume_du_muffin"

EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
TOP_FIELDS_FOR_DOC = ("title_fr", "ingredients_fr")  # (optionnel: "directions_fr")

BATCH_SIZE = 256


def build_document(row) -> str:
    title = str(row.get("title_fr", "")).strip()
    ing = str(row.get("ingredients_fr", "")).strip()
    # Optionnel: ajouter un peu d'étapes pour améliorer la recherche
    # steps = str(row.get("directions_fr", "")).strip()
    # return f"{title}\nIngrédients:\n{ing}\n\nPréparation:\n{steps}"
    return f"{title}\nIngrédients:\n{ing}"


def main():
    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(
            f"Fichier introuvable: {PROCESSED_PATH}\n"
            "Lance d'abord: python scripts/etl_recipenlg.py"
        )

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    print("📥 Chargement muffins FR:", PROCESSED_PATH)
    df = pd.read_parquet(PROCESSED_PATH)
    print(f"📦 Lignes: {len(df):,}")

    print("🤖 Chargement modèle embeddings:", EMBEDDING_MODEL_NAME)
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # Chroma persistant
    client = chromadb.Client(
        ChromaSettings(is_persistent=True, persist_directory=str(CHROMA_DIR))
    )

    # Recréer collection pour repartir propre
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    docs = [build_document(row) for _, row in df.iterrows()]
    metas = df.to_dict(orient="records")

    print("⚡ Indexation dans Chroma...")
    for start in tqdm(range(0, len(docs), BATCH_SIZE), desc="Batch"):
        end = min(start + BATCH_SIZE, len(docs))
        batch_docs = docs[start:end]
        batch_metas = metas[start:end]

        embeddings = model.encode(batch_docs, show_progress_bar=False).tolist()
        ids = [str(uuid.uuid4()) for _ in range(len(batch_docs))]

        collection.add(
            documents=batch_docs,
            embeddings=embeddings,
            metadatas=batch_metas,
            ids=ids
        )

    print(f"✅ Terminé. Recettes indexées: {collection.count():,}")
    print(f"📁 Chroma persist dir: {CHROMA_DIR}")


if __name__ == "__main__":
    main()
