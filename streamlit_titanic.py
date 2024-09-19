import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
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

# Charger les différents modèles
logreg_model = load_model('models/logreg.pkl')
rf_model = load_model('models/rf.pkl')
xgb_model = load_model('models/xgb.pkl')
knn_model = load_model('models/knn.pkl')

# Interface utilisateur Streamlit
st.title("Prédiction de la survie des passagers du Titanic")

# Sélection du modèle par l'utilisateur
model_choice = st.selectbox("Choisissez un modèle pour faire des prédictions", 
                            ("Régression Logistique", "Forêt Aléatoire", "XGBoost", "KNN"))

# Entrée des caractéristiques des passagers
age = st.slider("Âge du passager", 0, 80, 30)
fare = st.slider("Tarif du billet", 0, 500, 50)
gender = st.selectbox("Sexe", ["male", "female"])
embarked = st.selectbox("Port d'embarquement", ["S", "C", "Q"])
pclass = st.selectbox("Classe", [1, 2, 3])

# Prétraiter les données entrées par l'utilisateur
gender_num = 1 if gender == 'male' else 0
embarked_num = 0 if embarked == 'S' else (1 if embarked == 'C' else 2)

# Assemblage des caractéristiques
user_data = pd.DataFrame([[pclass, age, fare, gender_num, embarked_num]],
                         columns=["Pclass", "Age", "Fare", "Sex", "Embarked"])

# Normalisation si nécessaire
scaler = StandardScaler()
user_data_scaled = scaler.fit_transform(user_data)

# Faire des prédictions en fonction du modèle choisi
if model_choice == "Régression Logistique":
    prediction = logreg_model.predict(user_data_scaled)
elif model_choice == "Forêt Aléatoire":
    prediction = rf_model.predict(user_data_scaled)
elif model_choice == "XGBoost":
    prediction = xgb_model.predict(user_data_scaled)
else:
    prediction = knn_model.predict(user_data_scaled)

# Afficher le résultat
if prediction[0] == 1:
    st.success("Le passager a survécu")
else:
    st.error("Le passager n'a pas survécu")
