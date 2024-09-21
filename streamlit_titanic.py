#import pandas as pd
#import pickle
#import numpy as np

import streamlit as st
from PIL import Image

# Configuration de la page
st.set_page_config(page_title="Projet Titanic - Portfolio Data Analyst", layout="wide")

# Titre principal
st.title("Analyse des Données des Passagers du Titanic")
st.write("""
Bienvenue dans ce projet d'analyse des données sur les passagers du Titanic. Ce projet explore les différents facteurs qui ont influencé la survie des passagers lors du naufrage tragique en 1912. 
Nous allons parcourir différentes analyses statistiques et modélisations pour comprendre l'impact de variables comme l'âge, le sexe, la classe sociale, et la nationalité.
""")

# Section Introduction
st.header("Introduction")
st.write("""
Le naufrage du Titanic a causé la mort de plus de 1500 personnes. Ce projet analyse les données des passagers afin de comprendre quels facteurs ont influencé leurs chances de survie.
""")
st.write("""
Les résultats incluent des analyses statistiques et des modèles prédictifs, comme la régression logistique, Random Forest, XGBoost, et un classificateur Voting. Nous avons aussi utilisé le test du Chi² pour évaluer l'association entre certaines variables catégorielles et la survie.
""")

# Charger l'image du Titanic
image = Image.open('images/photo_titanic.jpg') 

# Afficher l'image dans l'application Streamlit
st.image(image, caption='Le Titanic', use_column_width=True)

# Section sur la distribution des données
st.header("Analyse Exploratoire des Données")
st.subheader("Distribution des variables")
st.write("""
L'analyse exploratoire a révélé plusieurs tendances intéressantes :
- L'âge des passagers est concentré majoritairement entre 20 et 40 ans.
- La majorité des passagers voyageaient en troisième classe.
- Les hommes représentaient environ 75 % des passagers.
""")

# Affichage des graphiques de distribution
st.subheader("Distribution des âges des passagers")
image_ages = Image.open("images/4 Distribution des âges des passagers.png")
st.image(image_ages, caption="Distribution des âges", use_column_width=True)

st.subheader("Répartition des passagers par sexe")
image_sexe = Image.open("images/5 Répartition des passagers par sexe.png")
st.image(image_sexe, caption="Répartition des passagers par sexe", use_column_width=True)

st.subheader("Répartition des passagers par classe")
image_classe = Image.open("images/6 Répartition des passagers par classe.png")
st.image(image_classe, caption="Répartition des passagers par classe", use_column_width=True)

# Section sur les corrélations
st.header("Corrélations des Variables")
st.write("""
Nous avons examiné les corrélations entre différentes variables pour mieux comprendre les facteurs influençant la survie des passagers. Le tarif du billet (fare) montre une corrélation positive avec la survie, tandis que le sexe et la classe sociale sont des facteurs clés.
""")
image_correlation = Image.open("images/11 Matrice de corrélation des variables numériques.png")
st.image(image_correlation, caption="Matrice de corrélation des variables numériques", use_column_width=True)

# Section sur les modèles prédictifs
st.header("Modélisation et Prédiction")
st.write("""
Plusieurs modèles ont été testés pour prédire la survie des passagers. Voici les résultats des principaux modèles :
- Régression Logistique
- Random Forest
- XGBoost
- Classificateur de Vote
""")

# Affichage des résultats des modèles
st.subheader("Courbes ROC des différents modèles")
st.write("Courbe ROC pour Random Forest, XGBoost et Voting Classifier.")
col1, col2, col3 = st.columns(3)
with col1:
    image_roc_rf = Image.open("images/21 courbe ROC  RF.png")
    st.image(image_roc_rf, caption="ROC Random Forest")
with col2:
    image_roc_xgb = Image.open("images/22 courbe ROC  XGB.png")
    st.image(image_roc_xgb, caption="ROC XGBoost")
with col3:
    image_roc_voting = Image.open("images/23 courbe ROC Voting classifier.png")
    st.image(image_roc_voting, caption="ROC Voting Classifier")

# Section sur l'importance des variables
st.header("Importance des Variables")
st.write("""
Le modèle Random Forest a permis d'identifier les variables les plus importantes dans la prédiction de la survie des passagers. Le sexe, l'âge et le tarif du billet sont les trois variables les plus influentes.
""")
image_importance = Image.open("images/25 Importance des 10 principales variables selon Random Forest.png")
st.image(image_importance, caption="Importance des variables selon Random Forest", use_column_width=True)

# Section Conclusion
st.header("Conclusion")
st.write("""
L'analyse approfondie des données des passagers du Titanic révèle que plusieurs facteurs ont influencé la survie :
- Les femmes et les enfants avaient de meilleures chances de survie.
- Les passagers de première classe avaient un avantage significatif.
- La nationalité et la langue parlée à bord ont également joué un rôle non négligeable.
""")
st.write("""
Les modèles prédictifs comme Random Forest et XGBoost ont montré des performances solides avec des AUC proches de 0,80. Ces résultats mettent en évidence l'importance de comprendre les interactions entre différentes variables lors de l'analyse de la survie dans de telles tragédies.
""")

st.write("Merci d'avoir consulté ce projet ! Pour plus de détails, consultez mon portfolio GitHub pour le code complet et les analyses supplémentaires.")




