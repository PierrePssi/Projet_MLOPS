# Utiliser une image Python officielle comme base
FROM python:3.9-slim

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 libsm6 libxext6 libxrender-dev

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier les fichiers nécessaires dans le conteneur
COPY . /app

# Installer les bibliothèques Python requises
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port pour l'application Streamlit
EXPOSE 8501

# Commande pour démarrer l'application Streamlit
CMD ["streamlit", "run", "app.py"]
