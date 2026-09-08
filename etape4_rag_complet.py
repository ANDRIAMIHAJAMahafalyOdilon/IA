"""
Étape 4+ : RAG optionnel + recherche web + analyse d'image.
Pour les questions texte simples, Gemini ET Groq répondent, puis une IA
arbitre choisit ou fusionne la meilleure réponse. Pour les images et la
recherche web, seul Gemini est utilisé (Groq ne les gère pas).
"""
import os
import base64
from dotenv import load_dotenv
from google import genai
from groq import Groq

from etape3_chunking_embeddings import (
    charger_index,
    construire_index,
    decouper_en_chunks,
    rechercher,
    sauvegarder_index,
    DOSSIER_INDEX,
)
from etape2_extraction_pdf import extraire_texte_pdf

load_dotenv()

client_gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
client_groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODELE_GEMINI = "gemini-3.6-flash"
MODELE_GROQ = "llama-3.3-70b-versatile"


def construire_texte_avec_contexte(question: str, passages: list[dict]) -> str:
    """Construit le texte du prompt, avec ou sans passages du PDF."""
    if not passages:
        return question

    contexte = "\n\n".join(
        f"[Page {p['page']}]\n{p['texte']}" for p in passages
    )

    return f"""Tu es un assistant d'étude qui aide un élève à comprendre son cours
et à faire ses devoirs.

Voici des passages extraits de son document de cours, qui semblent liés
à sa question :

--- PASSAGES DU COURS ---
{contexte}

--- QUESTION ---
{question}

Consignes pour ta réponse :
1. Utilise EN PRIORITÉ les passages du cours ci-dessus s'ils répondent à
   la question. Dans ce cas, cite les pages utilisées à la fin, sous la
   forme : Sources (document) : page X, page Y.
2. Si les passages ne suffisent pas, complète avec tes connaissances
   générales.
3. Sois toujours honnête sur l'origine de l'information.
4. Réponds de façon pédagogique, claire et structurée.
"""


def est_erreur_quota(exc: Exception) -> bool:
    """Détecte si l'erreur correspond à un dépassement de quota/rate limit."""
    message = str(exc).lower()
    return "429" in message or "quota" in message or "rate limit" in message


def repondre_avec_gemini(
    texte: str,
    image_data: bytes = None,
    image_mime: str = None,
    utiliser_web: bool = False,
) -> str:
    """Appelle Gemini, avec support image et recherche web."""
    input_parts = [{"type": "text", "text": texte}]

    if image_data is not None:
        input_parts.append({
            "type": "image",
            "data": base64.b64encode(image_data).decode("utf-8"),
            "mime_type": image_mime or "image/jpeg",
        })

    kwargs = {"model": MODELE_GEMINI, "input": input_parts}
    if utiliser_web:
        kwargs["tools"] = [{"type": "google_search"}]

    interaction = client_gemini.interactions.create(**kwargs)
    return interaction.output_text


def repondre_avec_groq(texte: str) -> str:
    """Texte uniquement (pas d'image, pas de web)."""
    completion = client_groq.chat.completions.create(
        model=MODELE_GROQ,
        messages=[{"role": "user", "content": texte}],
    )
    return completion.choices[0].message.content


def construire_prompt_jugement(question: str, reponse_a: str, reponse_b: str) -> str:
    """
    Demande à une IA d'arbitrer entre deux réponses et de produire
    la meilleure réponse finale possible pour l'élève.
    """
    return f"""Tu es un correcteur pédagogique exigeant. Un élève a posé la
question suivante, et deux assistants IA différents y ont répondu.

--- QUESTION DE L'ÉLÈVE ---
{question}

--- RÉPONSE DE L'ASSISTANT A ---
{reponse_a}

--- RÉPONSE DE L'ASSISTANT B ---
{reponse_b}

Ta tâche :
1. Compare les deux réponses : exactitude, clarté, pédagogie, complétude.
2. Produis UNE SEULE réponse finale, la meilleure possible pour l'élève —
   soit en gardant la meilleure des deux telle quelle, soit en fusionnant
   les points forts de chacune si c'est pertinent.
3. Ne mentionne PAS "assistant A" ou "assistant B" dans ta réponse finale
   — l'élève ne doit voir qu'une réponse propre et unifiée, comme si elle
   venait d'un seul expert.
4. Garde les éventuelles citations de pages du document si elles étaient
   présentes et correctes dans l'une des deux réponses.

Réponse finale :
"""


def repondre_comparaison(texte: str, question: str) -> str:
    """
    Fait répondre Gemini ET Groq à la même question, puis arbitre.
    Gère les échecs partiels (un seul des deux disponibles) sans planter.
    """
    reponse_gemini = None
    reponse_groq = None
    erreur_gemini = None
    erreur_groq = None

    try:
        reponse_gemini = repondre_avec_gemini(texte)
    except Exception as exc:
        erreur_gemini = exc

    try:
        reponse_groq = repondre_avec_groq(texte)
    except Exception as exc:
        erreur_groq = exc

    # Cas 1 : les deux ont répondu → on arbitre
    if reponse_gemini and reponse_groq:
        prompt_jugement = construire_prompt_jugement(question, reponse_gemini, reponse_groq)
        try:
            # On tente Gemini comme juge, puis Groq si Gemini est indisponible
            return repondre_avec_gemini(prompt_jugement)
        except Exception:
            try:
                return repondre_avec_groq(prompt_jugement)
            except Exception:
                # Si même le jugement échoue, on renvoie la réponse Gemini par défaut
                return reponse_gemini

    # Cas 2 : un seul a répondu → on l'utilise, avec une mention
    if reponse_gemini:
        return f"{reponse_gemini}\n\n*(Groq indisponible pour cette question, réponse de Gemini seul)*"
    if reponse_groq:
        return f"{reponse_groq}\n\n*(Gemini indisponible pour cette question, réponse de Groq seul)*"

    # Cas 3 : les deux ont échoué
    return f"⚠️ Gemini et Groq sont tous les deux indisponibles.\nGemini : {erreur_gemini}\nGroq : {erreur_groq}"


def repondre(
    question: str,
    index=None,
    chunks=None,
    k: int = 5,
    image_data: bytes = None,
    image_mime: str = None,
    utiliser_web: bool = False,
) -> str:
    """
    Pipeline principal :
    - Si image ou recherche web demandée → Gemini uniquement (Groq ne gère pas ça)
    - Sinon → comparaison Gemini + Groq, avec arbitrage automatique
    """
    passages = []
    if index is not None and chunks:
        passages = rechercher(question, index, chunks, k=k)

    texte = construire_texte_avec_contexte(question, passages)

    if image_data is not None or utiliser_web:
        try:
            return repondre_avec_gemini(
                texte, image_data=image_data, image_mime=image_mime, utiliser_web=utiliser_web
            )
        except Exception as exc:
            if est_erreur_quota(exc):
                return (
                    "⚠️ Quota Gemini dépassé. L'analyse d'image et la recherche "
                    "web ne sont disponibles que via Gemini — réessaie dans "
                    "une minute."
                )
            raise

    return repondre_comparaison(texte, question)


def obtenir_index(chemin_pdf: str):
    """Charge l'index existant, ou le construit s'il n'existe pas encore."""
    if (DOSSIER_INDEX / "index.faiss").exists():
        return charger_index()

    pages = extraire_texte_pdf(chemin_pdf)
    chunks = decouper_en_chunks(pages)
    index = construire_index(chunks)
    sauvegarder_index(index, chunks)
    return index, chunks


def main():
    print("Assistant en ligne de commande (comparaison Gemini + Groq)")
    print("Tape 'quitter' pour arrêter.\n")

    while True:
        question = input("Ta question : ").strip()
        if question.lower() in ("quitter", "exit", "q"):
            break
        if not question:
            continue

        reponse = repondre(question)
        print(f"\n🤖 {reponse}\n")


if __name__ == "__main__":
    main()