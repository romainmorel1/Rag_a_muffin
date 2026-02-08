import os
import re
import ast
import time
import random
from pathlib import Path

import pandas as pd
import kagglehub
from tqdm import tqdm
from deep_translator import GoogleTranslator
from deep_translator.exceptions import TranslationNotFound, TooManyRequests
from dotenv import load_dotenv

import requests
from requests.exceptions import ConnectionError, ReadTimeout


# =========================
# CONFIG
# =========================
load_dotenv()

DATASET_REF = "paultimothymooney/recipenlg"

OUT_PATH = Path("data/processed/muffins_fr.parquet")
CACHE_PATH = Path("data/processed/translation_cache.csv")

# Filtre muffin en ANGLAIS (plus fiable sur RecipeNLG)
MUFFIN_REGEX = re.compile(r"\b(?:muffin|muffins|cupcake|cupcakes)\b", re.IGNORECASE)

# Limite optionnelle pour prototyper vite (ex: 2000). Mets 0 pour tout.
LIMIT_MUFFINS = int(os.getenv("LIMIT_MUFFINS", "0")) or None

# Par défaut: on NE traduit PAS les directions (rapide & robuste)
TRANSLATE_DIRECTIONS = os.getenv("TRANSLATE_DIRECTIONS", "0").strip() in ("1", "true", "True", "yes", "YES")

# Traduction: pour éviter des erreurs sur textes longs, on coupe en chunks
MAX_CHARS_PER_CHUNK = 1500

TQDM_DESC = "Traduction FR"


# =========================
# HELPERS
# =========================
def find_main_file(dataset_dir: str) -> Path:
    """Trouve un fichier dataset exploitable (csv/json) dans le dossier KaggleHub."""
    p = Path(dataset_dir)
    candidates = list(p.rglob("*.csv")) + list(p.rglob("*.json")) + list(p.rglob("*.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"Aucun fichier .csv/.json/.jsonl trouvé dans {dataset_dir}")
    candidates.sort(key=lambda x: x.stat().st_size, reverse=True)
    return candidates[0]


def safe_literal_list(x):
    """
    RecipeNLG stocke souvent ingredients/directions comme des listes sérialisées en string.
    On convertit en list[str].
    """
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    if isinstance(x, list):
        return [str(i) for i in x]
    if isinstance(x, str):
        s = x.strip()
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
            try:
                v = ast.literal_eval(s)
                if isinstance(v, list):
                    return [str(i) for i in v]
            except Exception:
                pass
        return [s]
    return [str(x)]


def join_list_as_text(items, sep="\n- "):
    """Convertit une liste d'items en texte lisible pour embeddings."""
    if not items:
        return ""
    return "- " + sep.join([i.strip() for i in items if str(i).strip()])


def chunk_text(text: str, max_chars: int = MAX_CHARS_PER_CHUNK):
    """Découpe un texte long en chunks pour réduire les erreurs/rate limits."""
    if not text:
        return []
    text = str(text)
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end])
        start = end
    return chunks


def load_cache() -> dict:
    """Charge un cache de traduction: src -> fr."""
    if not CACHE_PATH.exists():
        return {}
    dfc = pd.read_csv(CACHE_PATH)
    if "src" not in dfc.columns or "fr" not in dfc.columns:
        return {}
    # En cas de doublons, garder le dernier
    return dict(zip(dfc["src"].astype(str), dfc["fr"].astype(str)))


def save_cache(cache: dict):
    """Sauve le cache sur disque."""
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"src": list(cache.keys()), "fr": list(cache.values())}).to_csv(CACHE_PATH, index=False)


def translate_fr_robust(text: str, translator: GoogleTranslator, *, retries: int = 5) -> str:
    """
    Traduction robuste:
    - retries + backoff
    - fallback EN en cas d'échec
    => ne lève jamais d'exception
    """
    if not text or not str(text).strip():
        return ""

    parts = chunk_text(str(text))
    out = []

    for part in parts:
        part = part.strip()
        if not part:
            continue

        translated = None
        for attempt in range(retries):
            try:
                translated = translator.translate(part)
                break
            except TranslationNotFound:
                # Inutile d'insister, on fallback
                translated = None
                break
            except TooManyRequests:
                # backoff
                time.sleep(1.5 * (attempt + 1) + random.random())
            except (ConnectionError, ReadTimeout, requests.exceptions.RequestException):
                # réseau / remote disconnected / anti-bot
                time.sleep(1.5 * (attempt + 1) + random.random())
            except Exception:
                time.sleep(1.5 * (attempt + 1) + random.random())

        out.append(translated if translated else part)  # fallback EN

    return "".join(out)


def translate_with_cache(text: str, translator: GoogleTranslator, cache: dict) -> str:
    """Traduit en utilisant un cache persistant pour éviter de retraduire."""
    t = str(text) if text is not None else ""
    if not t.strip():
        return ""
    if t in cache:
        return cache[t]
    fr = translate_fr_robust(t, translator)
    cache[t] = fr
    return fr


# =========================
# MAIN
# =========================
def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("⬇️  Téléchargement via KaggleHub (cache local)...")
    dataset_dir = kagglehub.dataset_download(DATASET_REF)
    print("📁 Path to dataset files:", dataset_dir)

    data_file = find_main_file(dataset_dir)
    print("📄 Fichier détecté:", data_file)

    print("📥 Chargement dataset...")
    if data_file.suffix.lower() == ".csv":
        df = pd.read_csv(data_file)
    elif data_file.suffix.lower() in [".json", ".jsonl"]:
        df = pd.read_json(data_file, lines=(data_file.suffix.lower() == ".jsonl"))
    else:
        raise ValueError(f"Format non supporté: {data_file.suffix}")

    print(f"📌 Lignes brutes: {len(df):,}")

    # Filtre muffins (EN)
    df["title"] = df["title"].fillna("").astype(str)
    mask = df["title"].str.contains(MUFFIN_REGEX, na=False)
    df_m = df.loc[mask].copy()
    print(f"🧁 Muffins détectés: {len(df_m):,}")

    if LIMIT_MUFFINS:
        df_m = df_m.sample(n=LIMIT_MUFFINS, random_state=42).copy()

        # df_m = df_m.head(LIMIT_MUFFINS).copy()
        print(f"🧪 LIMIT_MUFFINS activé: {len(df_m):,}")

    # Ingredients -> texte
    df_m["ingredients_list"] = df_m["ingredients"].apply(safe_literal_list)
    df_m["ingredients_text_en"] = df_m["ingredients_list"].apply(join_list_as_text)

    # Directions: on évite par défaut (trop long)
    if TRANSLATE_DIRECTIONS:
        df_m["directions_list"] = df_m["directions"].apply(safe_literal_list)
        df_m["directions_text_en"] = df_m["directions_list"].apply(join_list_as_text)
    else:
        df_m["directions_text_en"] = ""

    # Traduction FR (uniquement muffins)
    translator = GoogleTranslator(source="en", target="fr")
    cache = load_cache()
    tqdm.pandas(desc=TQDM_DESC)

    df_m["title_fr"] = df_m["title"].progress_apply(lambda x: translate_with_cache(x, translator, cache))
    save_cache(cache)  # checkpoint

    df_m["ingredients_fr"] = df_m["ingredients_text_en"].progress_apply(lambda x: translate_with_cache(x, translator, cache))
    save_cache(cache)  # checkpoint

    if TRANSLATE_DIRECTIONS:
        df_m["directions_fr"] = df_m["directions_text_en"].progress_apply(lambda x: translate_with_cache(x, translator, cache))
        save_cache(cache)
    else:
        df_m["directions_fr"] = ""  # pas de traduction des étapes

    # Dataset final
    keep_cols = [
        "#", "title", "ingredients", "directions", "link", "source", "NER",
        "title_fr", "ingredients_fr", "directions_fr"
    ]
    keep_cols = [c for c in keep_cols if c in df_m.columns]
    out = df_m[keep_cols].copy()

    # Nettoyage léger
    out["title_fr"] = out["title_fr"].fillna("").astype(str).str.strip()
    out["ingredients_fr"] = out["ingredients_fr"].fillna("").astype(str).str.strip()
    out["directions_fr"] = out["directions_fr"].fillna("").astype(str).str.strip()

    # Retirer les entrées vides (au minimum titre)
    out = out[out["title_fr"].str.len() > 0].reset_index(drop=True)

    print(f"💾 Sauvegarde: {OUT_PATH}")
    out.to_parquet(OUT_PATH, index=False)

    print("✅ ETL terminé.")
    print(f"📦 Lignes finales: {len(out):,}")
    print("🔎 Exemple:\n", out[["title_fr", "ingredients_fr"]].head(1).to_string(index=False))
    print(f"🗃️ Cache traduction: {CACHE_PATH} (entrées: {len(cache):,})")
    print(f"ℹ️ TRANSLATE_DIRECTIONS={TRANSLATE_DIRECTIONS}")


if __name__ == "__main__":
    main()
