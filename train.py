#  Importation des bibliothèques
import pandas as pd
import numpy as np
import os
import pickle
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.validation import check_is_fitted
from sklearn.linear_model import LogisticRegression
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

# Télécharger les ressources de nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# ---------------------------------- CHARGEMENT DES DONNÉES ---------------------------------- #

data_path = "all_film_reviews.csv"
df = pd.read_csv(data_path, encoding="ISO-8859-1", sep=";")

print("Aperçu des données :")
print(df.head())

# ---------------------------------- PRÉTRAITEMENT DES DONNÉES ---------------------------------- #

# supprimer les valeurs manquantes
df.dropna(inplace=True)

#  Conversion la colonne "Note" en float et normaliser le format
df["Note"] = df["Note"].astype(str).str.replace(",", ".").astype(float)

# Fonction pour attribuer un sentiment à chaque note
def assigner_sentiment(note):
    if pd.isna(note):
        return "Inconnu"
    elif note >= 4:
        return "Positif"
    else:
        return "Négatif"

# Ajouter la colonne "Sentiment" si elle n'existe pas
if "Sentiment" not in df.columns:
    df["Sentiment"] = df["Note"].apply(assigner_sentiment)


print("Répartition des sentiments :")
print(df["Sentiment"].value_counts())

# Tokenization, suppression des stopwords et lemmatisation
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("french"))
tokenizer = RegexpTokenizer(r'\w+')

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation)) 
    words = tokenizer.tokenize(text)  # Tokenization
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]  # Lemmatisation et suppression des stopwords
    return " ".join(words)


df["Description"] = df["Description"].apply(preprocess_text)


# ---------------------------------- ENTRAÎNEMENT DU TF-IDF Vectorizer ---------------------------------- #

vectorizer_path = "tfidf_vectorizer.pkl"

# Vérifier si un vectorizer existe déjà
if os.path.exists(vectorizer_path):
    try:
        vectorizer = joblib.load(vectorizer_path)
        print("Vectorizer chargé depuis le fichier existant.")
        
        if not hasattr(vectorizer, "vocabulary_"):
            raise AttributeError("Vectorizer non entraîné, réentraînement nécessaire.")

    except Exception as e:
        print(f"Problème avec le vectorizer existant ({e}), réentraînement en cours...")
        vectorizer = TfidfVectorizer(max_features=5000)
        X = vectorizer.fit_transform(df["Description"])
        joblib.dump(vectorizer, vectorizer_path)
        print("Nouveau vectorizer entraîné et sauvegardé.")
else:
    print("Aucun vectorizer trouvé, entraînement en cours...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df["Description"])
    joblib.dump(vectorizer, vectorizer_path)
    print("Nouveau vectorizer entraîné et sauvegardé.")

X = vectorizer.transform(df["Description"])


# -------------------------------------------- DIVISION DES DONNÉES EN TRAIN/TEST ---------------------------------- #


X_train, X_test, y_train, y_test = train_test_split(X, df["Sentiment"], test_size=0.2, random_state=42)

print("Données préparées, prêtes pour l'entraînement du modèle !")




#------------------------------------ ENTRAINEMENT DU MODELE------------------------------------#


model_path = "model_lr.pkl"

try:
    model = joblib.load(model_path)
    print("Modèle existant chargé.")
except FileNotFoundError:
    print("Aucun modèle trouvé, entraînement en cours...")
    model = LogisticRegression()
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)
    print("Modèle entraîné et sauvegardé.")

print("Modèle prêt à être utilisé !")


#----------------------------------------------- MLFLOW -------------------------------------------------#

mlflow.set_tracking_uri("http://127.0.0.1:8080")  # mlflow server --host 127.0.0.1 --port 8080
client = MlflowClient()

# Définition de l'expérience MLflow
experiment_name = "Sentiment_Analysis_Oscars"
experiment_description = (
    "Ce projet analyse les sentiments à partir des commentaires Allocinés "
    "sur les films nominés aux Oscars entre 2020 et 2024. "
    "L'objectif est de comprendre les tendances des notes à partir des commentaires."
)

experiment_tags = {
    "project_name": "analyse-sentiments",
    "movie_awards": "Oscars",
    "timeframe": "2020-2024",
    "team": "analyse-sentiments-team",
    "mlflow.note.content": experiment_description,
    "objective": "Classification de texte, Analyse de sentiments"
}

# Vérifier si l'expérience existe, sinon la créer
try:
    experiment_id = client.create_experiment(name=experiment_name, tags=experiment_tags)
except:
    experiment_id = client.get_experiment_by_name(experiment_name).experiment_id

mlflow.set_experiment(experiment_name)

# Démarrer un run MLflow
with mlflow.start_run():
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("vectorizer", "TF-IDF (max_features=5000)")

#------------------------------------ METRIQUES AVEC MLFLOW---------------------------------#
    
# Évaluer le modèle et enregistrer les métriques
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)

mlflow.log_metric("train_accuracy", train_accuracy)
mlflow.log_metric("test_accuracy", test_accuracy)

print(f"Précision sur les données d'entraînement : {train_accuracy:.4f}")
print(f"Précision sur les données de test : {test_accuracy:.4f}")

# Enregistrer le modèle avec MLflow
mlflow.sklearn.log_model(model, "logistic_regression_model")
print("Modèle sauvegardé avec MLflow.")

mlflow.sklearn.log_model(vectorizer, "tfidf_vectorizer")
print("Vectorizer sauvegardé avec MLflow.")
