import streamlit as st
import pickle
import numpy as np

# Charger le modèle et le vectorizer
with open("model_lr.pkl", "rb") as model_file:
    model_lr = pickle.load(model_file)
with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

# Interface Streamlit
st.title("Analyse de Sentiment des Avis de Films")
st.write("Entrez un avis et le modèle prédira s'il est positif ou négatif.")

# Champ de texte pour l'entrée utilisateur
user_input = st.text_area("Votre avis :", "")

if st.button("Prédire"):
    if user_input:
        # Transformer l'entrée utilisateur
        user_input_tfidf = tfidf_vectorizer.transform([user_input])
        
        # Prédiction
        prediction = model_lr.predict(user_input_tfidf)[0]
        
        # Affichage du résultat
        sentiment = "Positif" if prediction == 1 else "Négatif"
        st.write(f"### Sentiment prédit : {sentiment}")
    else:
        st.warning("Veuillez entrer un texte avant de prédire.")
