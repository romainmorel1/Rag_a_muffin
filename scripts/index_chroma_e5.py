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
COLLECTION_NAME = "royaume_du_muffin_e5"

MODEL_NAME = "intfloat/multilingual-e5-small"
BATCH_SIZE = 256


def build_document(row) -> str:
    """
    Document 'passage' pour E5.
    On ajoute NER pour renforcer les ingrédients (souvent plus stable que les traductions).
    """
    title = str(row.get("title_fr", "")).strip()
    ing = str(row.get("ingredients_fr", "")).strip()
    ner = str(row.get("NER", "")).strip()

    # Note: E5 attend souvent les docs préfixés par "passage: "
    doc = f"{title}\nIngrédients:\n{ing}"
    if ner and ner != "nan":
        doc += f"\n\nNER:\n{ner}"

    return "passage: " + doc


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

    print("🤖 Chargement modèle embeddings:", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)

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

    print("⚡ Indexation dans Chroma (E5)...")
    for start in tqdm(range(0, len(docs), BATCH_SIZE), desc="Batch"):
        end = min(start + BATCH_SIZE, len(docs))
        batch_docs = docs[start:end]
        batch_metas = metas[start:end]

        # E5: normaliser est une bonne pratique pour la similarité cosinus
        embeddings = model.encode(
            batch_docs,
            show_progress_bar=False,
            normalize_embeddings=True
        ).tolist()

        ids = [str(uuid.uuid4()) for _ in range(len(batch_docs))]

        collection.add(
            documents=batch_docs,
            embeddings=embeddings,
            metadatas=batch_metas,
            ids=ids
        )

    print(f"✅ Terminé. Recettes indexées: {collection.count():,}")
    print(f"📁 Chroma persist dir: {CHROMA_DIR}")
    print(f"📚 Collection: {COLLECTION_NAME}")


if __name__ == "__main__":
    main()
