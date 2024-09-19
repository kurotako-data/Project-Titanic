import pandas as pd
import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from PIL import Image

# Charger l'image
image = Image.open('images/photo_titanic.jpg') 

# Afficher l'image dans l'application Streamlit
st.image(image, caption='Le Titanic', use_column_width=True)

# Fonction pour charger les modèles
def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model


# Charger les modèles entraînés
with open('models/logreg.pkl', 'rb') as file:
    logreg_model = pickle.load(file)
with open('models/rf.pkl', 'rb') as file:
    rf_model = pickle.load(file)
with open('models/xgb.pkl', 'rb') as file:
    xgb_model = pickle.load(file)
with open('models/knn.pkl', 'rb') as file:
    knn_model = pickle.load(file)

# Fonction de prédiction avec le modèle sélectionné
def predict_survival(model, user_data):
    # Prétraitement des données comme pour l'entraînement (par exemple, encodage des variables catégorielles)
    # Variables encodées et standardisées ici
    categorical_features = ['sex', 'embarked', 'class']
    numerical_features = ['age', 'fare']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(), categorical_features)])

    user_data_scaled = preprocessor.fit_transform(user_data)
    
    # Prédiction
    prediction = model.predict(user_data_scaled)
    return prediction

# Interface Streamlit
st.title("Prédictions de survie du Titanic")

# Sélection du modèle
model_choice = st.selectbox("Choisissez un modèle pour faire des prédictions", 
                            ("Régression Logistique", "Forêt Aléatoire", "XGBoost", "KNN"))

# Collecter les entrées utilisateur
age = st.slider("Âge du passager", 0, 80, 30)
fare = st.slider("Tarif du billet", 0, 500, 50)
sex = st.selectbox("Sexe", ("male", "female"))
embarked = st.selectbox("Port d'embarquement", ("S", "C", "Q"))
pclass = st.selectbox("Classe", (1, 2, 3))

# Convertir les données en tableau pour prédiction
user_data = np.array([[age, fare, sex, embarked, pclass]])

# Choisir le modèle
if model_choice == "Régression Logistique":
    model = logreg_model
elif model_choice == "Forêt Aléatoire":
    model = rf_model
elif model_choice == "XGBoost":
    model = xgb_model
else:
    model = knn_model

# Bouton de prédiction
if st.button("Prédire la survie"):
    result = predict_survival(model, user_data)
    st.write(f"La prédiction est: {'Survivra' if result == 1 else 'Ne survivra pas'}")

