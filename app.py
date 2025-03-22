import streamlit as st
import joblib
#import pandas as pd

# Charger le modèle et le vectorizer
model_path = "model_lr.pkl"
vectorizer_path = "tfidf_vectorizer.pkl"

st.title("🎬 Analyse de Sentiments des Films")

try:
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    st.success("✅ Modèle et vectorizer chargés avec succès !")
except Exception as e:
    st.error(f"⚠️ Erreur lors du chargement des fichiers : {e}")

# Interface utilisateur
st.write("📌 Entrez une critique de film et obtenez son sentiment (Positif/Négatif).")

# Saisie utilisateur
user_input = st.text_area("📝 Écrivez votre critique ici :", "")

if st.button("Prédire"):
    if user_input:
        # Transformer le texte avec le vectorizer
        X_input = vectorizer.transform([user_input])
        prediction = model.predict(X_input)[0]
        
        # Affichage du résultat
        if prediction == "Positif":
            st.success("😃 Sentiment détecté : **Positif**")
        else:
            st.error("😞 Sentiment détecté : **Négatif**")
    else:
        st.warning("❗ Merci d'entrer un texte avant de prédire.")
