"""
Interface Streamlit pour l'AI Study Assistant.
PDF optionnel, recherche web activable, analyse d'image possible,
et gestion propre des erreurs de quota API.
"""
import streamlit as st
from pathlib import Path

from etape2_extraction_pdf import extraire_texte_pdf
from etape3_chunking_embeddings import decouper_en_chunks, construire_index
from etape4_rag_complet import repondre

st.set_page_config(page_title="AI Study Assistant", page_icon="📚")
st.title("📚 AI Study Assistant")
st.caption("Pose tes questions, avec ou sans document, avec ou sans image, avec ou sans recherche web.")

# --- Initialisation de l'état de session ---
if "historique" not in st.session_state:
    st.session_state.historique = []
if "index" not in st.session_state:
    st.session_state.index = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "nom_document" not in st.session_state:
    st.session_state.nom_document = None

# --- Barre latérale ---
with st.sidebar:
    st.header("⚙️ Options")

    st.subheader("📄 Document (optionnel)")
    fichier_pdf = st.file_uploader("Importe un PDF de cours", type=["pdf"])

    if fichier_pdf is not None and fichier_pdf.name != st.session_state.nom_document:
        with st.spinner("Analyse du PDF en cours..."):
            Path("documents").mkdir(exist_ok=True)
            chemin_temp = Path("documents") / fichier_pdf.name
            with open(chemin_temp, "wb") as f:
                f.write(fichier_pdf.getbuffer())

            pages = extraire_texte_pdf(str(chemin_temp))
            chunks = decouper_en_chunks(pages)
            index = construire_index(chunks)

            st.session_state.index = index
            st.session_state.chunks = chunks
            st.session_state.nom_document = fichier_pdf.name

        st.success(f"'{fichier_pdf.name}' indexé ({len(chunks)} chunks)")

    if st.session_state.nom_document:
        st.info(f"Document actif : {st.session_state.nom_document}")
        if st.button("❌ Retirer le document"):
            st.session_state.index = None
            st.session_state.chunks = None
            st.session_state.nom_document = None
            st.rerun()

    st.subheader("🔎 Recherche web")
    utiliser_web = st.checkbox("Autoriser Gemini à chercher sur le web")
    if utiliser_web:
        st.caption("⚠️ Chaque requête avec recherche web a un coût légèrement plus élevé.")

    st.subheader("🖼️ Image (optionnel)")
    fichier_image = st.file_uploader(
        "Joindre une image à ta prochaine question",
        type=["jpg", "jpeg", "png", "webp"],
        key="uploader_image",
    )

    if st.button("🗑️ Effacer l'historique"):
        st.session_state.historique = []
        st.rerun()

    st.divider()
    st.caption(
        "💡 Si tu vois une erreur de quota (429), attends une minute avant "
        "de reposer une question — c'est la limite du plan gratuit Gemini."
    )

# --- Zone principale : conversation ---
for echange in st.session_state.historique:
    with st.chat_message("user"):
        if echange.get("image"):
            st.image(echange["image"], width=200)
        st.write(echange["question"])
    with st.chat_message("assistant"):
        st.write(echange["reponse"])

question = st.chat_input("Pose ta question...")

if question:
    image_data = None
    image_mime = None
    image_affichage = None

    if fichier_image is not None:
        image_data = fichier_image.getvalue()
        image_mime = fichier_image.type
        image_affichage = image_data

    with st.chat_message("user"):
        if image_affichage:
            st.image(image_affichage, width=200)
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Réflexion en cours..."):
            try:
                reponse = repondre(
                    question,
                    index=st.session_state.index,
                    chunks=st.session_state.chunks,
                    k=5,
                    image_data=image_data,
                    image_mime=image_mime,
                    utiliser_web=utiliser_web,
                )
            except Exception as exc:
                message_erreur = str(exc)
                if "429" in message_erreur or "quota" in message_erreur.lower():
                    reponse = (
                        "⚠️ Limite de requêtes gratuites atteinte pour le moment. "
                        "Attends une minute et réessaie, ou consulte ton quota sur "
                        "https://ai.dev/rate-limit"
                    )
                else:
                    reponse = f"⚠️ Une erreur est survenue : {message_erreur}"
        st.write(reponse)

    st.session_state.historique.append({
        "question": question,
        "reponse": reponse,
        "image": image_affichage,
    })