# Projet RAG à Muffin 

Le but du projet est de développer un assistant culinaire spécialiser dans les recettes de muffin à partir d'un RAG (Retrieval-Augmented-Generation). 
L'utilisateur entre une requête où figure ses envies ou les ingrédients dont il dispose. 

L'assistant culinaire vise exclusivement à proposer des recettes de muffin. 
Si l'utilisateur formule une demande hors-sujet (recettes de pizza, gratin dauphinois, lasagne), l'assistant refuse explicititement de répondre, précisant qu'il n'est habilité qu'à proposer des recettes de muffin. 

Le projet a été réalisé entiérement en français.

## Structure du projet 

```text
Rag_a_muffin/
│
├── app/
│   └── streamlit_app.py        # Interface utilisateur (Streamlit)
│
├── src/
│   ├── __init__.py
│   ├── rag_app.py              # Logique RAG (retrieval + génération)
│   ├── llm_mistral.py          # Wrapper API Mistral (LLM)
│   ├── prompts.py              # System prompt (Chef Muffin)
│   └── utils.py                # Fonctions utilitaires (si besoin)
│
├── scripts/
│   ├── etl_recipenlg.py        # ETL : nettoyage, filtrage muffins, traduction
│   ├── index_chroma.py         # Indexation des recettes dans ChromaDB
│   ├── query_chroma.py         # Tests de recherche vectorielle
│   ├── query_chroma_e5_mmr.py  # Recherche avec embeddings E5 + MMR
│   └── compare_embeddings.py   # Comparaison et visualisation des embeddings
│
├── data/
│   ├── raw/                    # (optionnel) Données brutes
│   ├── processed/
│   │   └── muffins_fr.parquet  # Dataset final filtré (muffins uniquement)
│   └── chroma/                 # Base vectorielle persistée (ChromaDB)
│
├── experiments/
│   ├── embedding_analysis/     # Visualisations UMAP, analyses exploratoires
│   └── notes.md                # Observations, essais, constats
│
├── requirements.txt            # Dépendances Python
├── README.md                   # Documentation du projet
├── .gitignore                  # Exclusions Git (.env, data lourdes, etc.)
└── .env.example                # Exemple de configuration d’environnement
```

## Dataset 

Le dataset utilisé est RECIPENLG, disponible sur Kaggle. 
Il s’agit d’un corpus de grande ampleur regroupant environ 2,23 millions de recettes, issues de différentes sources culinaires.

Les principales colonnes du dataset sont :
- title : titre de la recette
- ingredients : liste d’ingrédients
- directions : instructions de préparation
- link : lien vers la recette originale
- source : source du contenu
- NER : entités extraites automatiquement

### Pré-traitement et filtrage de la donnée 

#### Filtrage thématique 

Afin de garantir que l’assistant ne propose que des muffins, un filtrage strict a été appliqué sur le champ title.
Les recettes ont été conservées uniquement si leur titre contenait l’un des mots-clés suivants (insensibles à la casse) :
- muffin
- muffins
- cupcake
- cupcakes
Ce choix volontairement simple et conservateur permet :
d’exclure efficacement les recettes hors-sujet,
de limiter les ambiguïtés sémantiques,
de garantir un domaine strictement contrôlé.
Après filtrage, environ 32 000 recettes liées aux muffins ont été identifiées.

#### Sous-échantillonage

Pour des raisons de temps de calcul (notamment liées à la traduction et à la vectorisation), un sous-ensemble de 3 000 recettes a été utilisé pour la phase finale d’indexation.
Ce choix permet :
- un prototypage rapide,
- des expérimentations itératives,
tout en conservant une diversité suffisante de recettes (sucrées et salées).

### Traduction et traitement linguistique

La traduction automatique a été réalisée à l’aide d’un service de traduction externe.
Afin de garantir la robustesse du pipeline, seules les parties suivantes ont été traduites :
- le titre de la recette,
- la liste des ingrédients.
Les instructions de préparation (```directions```) n’ont pas été traduites dans la version finale du projet.
Ce choix est motivé par plusieurs raisons :
- les textes d’instructions sont souvent longs et hétérogènes,
- la traduction automatique de textes longs s’est révélée fragile,
- la traduction des ingrédients est suffisante pour une recherche sémantique pertinente.
Ce compromis permet de limiter les erreurs tout en conservant une bonne qualité de retrieval.

## Vectorisation et choix du modèle d'embeddings 

### Modèles évalués 

Deux modèles d’embeddings multilingues ont été testés et comparés :
- ```paraphrase-multilingual-MiniLM-L12-v2```
- ```intfloat/multilingual-e5-small``

### Choix final

Après expérimentation, le modèle **E5 multilingue** (```intfloat/multilingual-e5-small```) a été retenu pour la version finale.
Ce choix est motivé par :
une meilleure prise en compte des requêtes en français,
une meilleure correspondance sémantique entre ingrédients et recettes,
des résultats plus cohérents sur des requêtes culinaires variées.
Les embeddings produits ont une dimension de **384** et sont **normalisés**, conformément aux recommandations du modèle E5.
Les formats ```query:``` et ```passage:``: ont été utilisés pour distinguer les requêtes utilisateur des documents indexés.
Une visualisation des embeddings (UMAP) a également été réalisée afin d’analyser la structure sémantique de l’espace vectoriel.

## Base vectorielle et stratégie de retrieval

Les embeddings ont été stockés dans une base vectorielle **ChromaDB**, avec persistance locale.
Pour la recherche, une stratégie de **MMR (Maximal Marginal Relevance)** a été utilisée.
Cette approche permet de :
- maximiser la similarité avec la requête utilisateur,
- tout en évitant des résultats trop redondants.
Concrètement :
- plusieurs recettes candidates sont récupérées (top-k),
- une seule recette finale est sélectionnée pour la génération,
- la diversité et la qualité du contexte sont améliorées.

## Génération de la réponse et prompt engineering 

La génération finale repose sur un **LLM accessible via l’API Mistral**.
Un prompt système strict définit le comportement de l’assistant, incarné par le persona **Chef Muffin**.
Les règles principales imposées par le prompt sont :
- proposer **une seule recette**,
- refuser explicitement toute demande hors-muffin,
- ne jamais inventer d’ingrédients ou d’étapes,
- utiliser uniquement les informations fournies dans le contexte récupéré,
- répondre en français, de manière claire et concise.
Un format de sortie strict est imposé afin d’obtenir une réponse structurée :
- nom de la recette,
- liste d’ingrédients,
- étapes de préparation.

À noteer que pour repreoduire le projet, il est nécessaire de disposer d'un compte API mistral permettant de générer une clé API. 

Cette clé API doit ensuite être renseignée en tant que variable d'environnement via la commande dans le terminal :
```md
export MISTRAL_API_KEY="..."
```

## Reproduction du projet 

Afin de pouvoir utiliser l'assistant, il est nécessaire de télécharger la donnée et de constituer la base vectorielle en suivant les instructions suivantes: 

### Installation de l'environnement 

Pour installer un environnement python capable de faire tourner l'assistant, appliquer les commandes suivantes:

```
bash
conda create -n rag-muffin python=3.10
conda activate rag-muffin
pip install -r requirements.txt

```
### Téléchargement et prétaitement des données 

La variable ```LIMIT_MUFFINS``` permet de limiter à 3000 recettes le nombre de recettes considérées pour constituer sa base de données vectorielles
```bash
export LIMIT_MUFFINS=3000
python scripts/etl_recipenlg.py
```
### Indexation vectorielle 

```bash

python scripts/index_chroma_e5.py
```

### Renseigner sa clé API mistral 

```md
export MISTRAL_API_KEY="..."
```

### Lancement de l'application via une application streamlit

```bash
export PYTHONPATH=.
streamlit run app/streamlit_app.py
```





