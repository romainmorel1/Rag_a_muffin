import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import uuid

# --- CONFIGURATION FRANÇAISE 🇫🇷 ---
# Modèle multilingue indispensable pour bien comprendre le français
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "royaume_du_muffin"

def load_and_simulate_data():
    """ Simule un chargement de données (à remplacer par vos données) """
    data = [
        {"title": "Muffins tout chocolat", "ingredients": "farine, sucre, oeufs, chocolat noir, beurre", "type": "sucré"},
        {"title": "Lasagnes à la bolognaise", "ingredients": "pâtes, boeuf, tomate, fromage, oignon", "type": "plat"},
        {"title": "Muffins salés chèvre épinards", "ingredients": "farine, oeufs, chèvre, épinards, huile d'olive", "type": "salé"},
        {"title": "Salade César", "ingredients": "laitue, poulet, parmesan, croutons", "type": "entrée"},
        {"title": "Cupcakes vanille (façon muffin)", "ingredients": "farine, sucre, vanille, glaçage", "type": "sucré"}
    ]
    return pd.DataFrame(data)

def filter_only_muffins(df):
    """ FILTRE STRICT : On ne garde que les muffins ! """
    keywords = ["muffin", "cupcake", "moelleux"]
    # Filtre insensible à la casse
    mask = df['title'].str.contains('|'.join(keywords), case=False, na=False)
    return df[mask]

def create_embeddings_and_store(df):
    print("🤖 Chargement du modèle d'embedding multilingue...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # Concaténation Titre + Ingrédients pour la recherche
    documents = (df['title'] + " : " + df['ingredients']).tolist()
    metadatas = df.to_dict(orient='records')
    ids = [str(uuid.uuid4()) for _ in range(len(df))]

    print("⚡ Vectorisation en cours...")
    embeddings = model.encode(documents).tolist()

    # Stockage ChromaDBP
    client = chromadb.Client() # En mémoire pour le test
    try: client.delete_collection(name=COLLECTION_NAME)
    except: pass

    collection = client.create_collection(name=COLLECTION_NAME)
    collection.add(documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids)

    print(f"✅ Indexation terminée ! {collection.count()} recettes stockées.")
    return collection

# --- TEST ---
if __name__ == "__main__":
    df = load_and_simulate_data()
    df_muffins = filter_only_muffins(df)
    db = create_embeddings_and_store(df_muffins)

    # Test de recherche en français
    query = "Je veux utiliser mes restes de fromage"
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    results = db.query(query_embeddings=model.encode([query]).tolist(), n_results=1)
    print(f"\n🔎 Question: '{query}'")
    print(f"👉 Réponse RAG: {results['metadatas'][0][0]['title']}")