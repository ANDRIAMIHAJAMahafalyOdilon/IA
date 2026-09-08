"""
Étape 3 : découpage du texte en chunks, génération des embeddings
avec Gemini, et recherche par similarité avec FAISS.
"""
import pickle
import os
from pathlib import Path

import numpy as np
import faiss
from google import genai
from dotenv import load_dotenv

from etape2_extraction_pdf import extraire_texte_pdf

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODELE_EMBEDDING = "gemini-embedding-001"
DOSSIER_INDEX = Path("index")
TAILLE_CHUNK = 400        # mots par chunk
CHEVAUCHEMENT = 50        # mots de chevauchement entre chunks


def decouper_en_chunks(pages_texte: list[dict]) -> list[dict]:
    """
    Découpe le texte de chaque page en chunks de ~TAILLE_CHUNK mots,
    avec un léger chevauchement pour ne pas couper une idée en deux.
    """
    chunks = []
    for page in pages_texte:
        mots = page["texte"].split()
        if not mots:
            continue

        debut = 0
        while debut < len(mots):
            fin = debut + TAILLE_CHUNK
            morceau = " ".join(mots[debut:fin])
            chunks.append({"page": page["page"], "texte": morceau})
            debut += TAILLE_CHUNK - CHEVAUCHEMENT

    return chunks


def generer_embedding(texte: str) -> list[float]:
    """Appelle l'API Gemini pour obtenir le vecteur d'un texte."""
    resultat = client.models.embed_content(
        model=MODELE_EMBEDDING,
        contents=texte,
    )
    return resultat.embeddings[0].values


def construire_index(chunks: list[dict]) -> faiss.Index:
    """
    Génère l'embedding de chaque chunk et construit un index FAISS
    permettant la recherche par similarité cosinus.
    """
    print(f"Génération des embeddings pour {len(chunks)} chunks...")
    vecteurs = []
    for i, chunk in enumerate(chunks, start=1):
        vecteur = generer_embedding(chunk["texte"])
        vecteurs.append(vecteur)
        print(f"  chunk {i}/{len(chunks)} traité", end="\r")

    print("\nConstruction de l'index FAISS...")
    matrice = np.array(vecteurs, dtype="float32")
    faiss.normalize_L2(matrice)

    index = faiss.IndexFlatIP(matrice.shape[1])
    index.add(matrice)
    return index


def sauvegarder_index(index: faiss.Index, chunks: list[dict]):
    """Sauvegarde l'index FAISS et les métadonnées des chunks sur disque."""
    DOSSIER_INDEX.mkdir(exist_ok=True)
    faiss.write_index(index, str(DOSSIER_INDEX / "index.faiss"))
    with open(DOSSIER_INDEX / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    print(f"Index sauvegardé dans {DOSSIER_INDEX}/")


def charger_index() -> tuple[faiss.Index, list[dict]]:
    """Recharge un index déjà construit, pour éviter de tout refaire."""
    index = faiss.read_index(str(DOSSIER_INDEX / "index.faiss"))
    with open(DOSSIER_INDEX / "chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


def rechercher(question: str, index: faiss.Index, chunks: list[dict], k: int = 3) -> list[dict]:
    """Retourne les k chunks les plus proches sémantiquement de la question."""
    vecteur_question = np.array([generer_embedding(question)], dtype="float32")
    faiss.normalize_L2(vecteur_question)

    scores, indices = index.search(vecteur_question, k)

    resultats = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        resultats.append({**chunks[idx], "score": float(score)})
    return resultats


def main():
    chemin_pdf = "documents/mon_cours.pdf"  # ← adapte si besoin
    index_existe = (DOSSIER_INDEX / "index.faiss").exists()

    if index_existe:
        print("Index existant trouvé, chargement...")
        index, chunks = charger_index()
    else:
        print(f"Extraction et indexation de : {chemin_pdf}")
        pages = extraire_texte_pdf(chemin_pdf)
        chunks = decouper_en_chunks(pages)
        index = construire_index(chunks)
        sauvegarder_index(index, chunks)

    print(f"\n{len(chunks)} chunks disponibles pour la recherche.\n")
    print("Tape 'quitter' pour arrêter.\n")

    while True:
        question = input("Question (recherche seulement, pas de réponse IA) : ").strip()
        if question.lower() in ("quitter", "exit", "q"):
            break
        if not question:
            continue

        resultats = rechercher(question, index, chunks, k=3)
        print(f"\n🔎 {len(resultats)} passage(s) pertinent(s) trouvé(s) :\n")
        for r in resultats:
            print(f"--- Page {r['page']} (score: {r['score']:.3f}) ---")
            print(r["texte"][:300] + "...\n")


if __name__ == "__main__":
    main()