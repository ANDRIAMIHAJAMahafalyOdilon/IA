"""
Étape 1 : script minimal qui envoie une question à Gemini
et affiche la réponse. Sert de base avant d'ajouter le PDF.
"""
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "Clé API introuvable. Vérifie que ton fichier .env contient "
        "bien GEMINI_API_KEY=ta-cle"
    )

client = genai.Client(api_key=api_key)

NOM_MODELE = "gemini-3.6-flash"


def poser_question(question: str) -> str:
    """Envoie une question à Gemini et retourne la réponse texte."""
    interaction = client.interactions.create(
        model=NOM_MODELE,
        input=question,
    )
    return interaction.output_text


def main():
    print("=== AI Study Assistant — Étape 1 ===")
    print("Tape 'quitter' pour arrêter.\n")

    while True:
        question = input("Ta question : ").strip()
        if question.lower() in ("quitter", "exit", "q"):
            print("À bientôt !")
            break
        if not question:
            continue

        try:
            reponse = poser_question(question)
            print(f"\nGemini : {reponse}\n")
        except Exception as exc:
            print(f"\n⚠️ Erreur lors de l'appel à l'API : {exc}\n")


if __name__ == "__main__":
    main()