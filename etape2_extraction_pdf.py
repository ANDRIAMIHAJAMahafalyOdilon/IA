"""
Étape 2 : extraction du texte brut d'un fichier PDF.
Prépare le terrain avant la recherche de passages pertinents (étape 3).
"""
from pathlib import Path
from pypdf import PdfReader


def extraire_texte_pdf(chemin_pdf: str) -> list[dict]:
    """
    Lit un PDF et retourne une liste de dictionnaires,
    un par page, avec le numéro de page et son texte.
    """
    chemin = Path(chemin_pdf)
    if not chemin.exists():
        raise FileNotFoundError(f"Fichier introuvable : {chemin_pdf}")

    lecteur = PdfReader(chemin)
    pages_texte = []

    for numero, page in enumerate(lecteur.pages, start=1):
        texte = page.extract_text() or ""
        pages_texte.append({"page": numero, "texte": texte.strip()})

    return pages_texte


def afficher_apercu(pages_texte: list[dict], nb_caracteres: int = 300):
    """Affiche un aperçu du texte extrait, page par page."""
    print(f"\n📄 {len(pages_texte)} page(s) extraite(s)\n")
    for p in pages_texte:
        apercu = p["texte"][:nb_caracteres]
        print(f"--- Page {p['page']} ---")
        print(apercu + ("..." if len(p["texte"]) > nb_caracteres else ""))
        print()


def main():
    chemin_pdf = "documents/mon_cours.pdf"  # ← adapte si besoin

    print(f"Extraction de : {chemin_pdf}")
    pages = extraire_texte_pdf(chemin_pdf)
    afficher_apercu(pages)


if __name__ == "__main__":
    main()