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
Bienvenue dans cette analyse approfondie des données des passagers du Titanic. Ce projet se concentre sur l'exploration des facteurs ayant influencé les chances de survie lors du naufrage tragique du Titanic en 1912.
Nous allons découvrir la composition des passagers en termes de sexe, d'âge, et de nationalité, ainsi que leurs chances de survie en fonction de ces facteurs. 
Enfin, nous examinerons la performance de plusieurs modèles prédictifs visant à estimer les probabilités de survie.
""")

# Section Introduction
st.header("Introduction")
st.write("""
Le naufrage du Titanic a été l'un des événements maritimes les plus tragiques de l'histoire, entraînant la mort de plus de 1500 personnes. Cette catastrophe a suscité de nombreuses questions sur les facteurs ayant influencé la survie des passagers. Ce projet s'efforce de répondre à certaines de ces questions en analysant les données des passagers.
""")
st.write("""
Les résultats de cette analyse mettent en lumière plusieurs facteurs clés tels que le sexe, l'âge, la classe sociale et la nationalité, et démontrent comment ces variables ont influencé les probabilités de survie des passagers.
""")

# Charger l'image du Titanic
image = Image.open('images/photo_titanic.jpg') 
st.image(image, caption='Le Titanic', use_column_width=True)

# Section sur la situation initiale des passagers
st.header("Situation Initiale des Passagers")
st.write("""
Nous commençons par explorer la répartition des passagers selon trois variables cruciales : le sexe, l'âge et la nationalité. Ces éléments nous permettent de mieux comprendre la composition des passagers à bord avant la tragédie.
""")

st.subheader("Répartition des passagers par sexe")
image_sexe = Image.open("images/5 Répartition des passagers par sexe.png")
st.image(image_sexe, caption="Répartition des passagers par sexe", use_column_width=True)
st.write("""
Environ 75 % des passagers étaient des hommes, tandis que les femmes représentaient une proportion plus faible. Cela aura un impact significatif sur les taux de survie, comme nous le verrons plus tard.
""")

st.subheader("Répartition des passagers par nationalité")
image_nationalite = Image.open("images/15 Répartition des passagers par nationalité (Top 10).png")
st.image(image_nationalite, caption="Répartition des passagers par nationalité (Top 10)", use_column_width=True)
st.write("""
La majorité des passagers à bord provenaient d'Angleterre, suivis par des passagers des États-Unis et d'Irlande. Cela reflète la diversité des passagers à bord du Titanic, ainsi que la répartition géographique des voyageurs transatlantiques à cette époque.
""")

st.subheader("Distribution des âges des passagers")
image_ages = Image.open("images/4 Distribution des âges des passagers.png")
st.image(image_ages, caption="Distribution des âges des passagers", use_column_width=True)
st.write("""
La plupart des passagers avaient entre 20 et 40 ans, avec une forte concentration de jeunes adultes. Il est important de noter que cette tranche d'âge était probablement plus active lors de l'évacuation, ce qui pourrait avoir eu un impact sur les taux de survie.
""")

# Section sur les survivants
st.header("Caractéristiques des Survivants")
st.write("""
Examinons à présent la composition des survivants, toujours selon les mêmes variables. Nous allons observer comment ces facteurs ont influencé les chances de survie.
""")

st.subheader("Taux de survie par sexe")
image_survie_sexe = Image.open("images/7 Taux de survie par sexe.png")
st.image(image_survie_sexe, caption="Taux de survie par sexe", use_column_width=True)
st.write("""
Les femmes avaient un taux de survie beaucoup plus élevé (environ 75 %) comparé aux hommes (environ 20 %). Cela s'explique en grande partie par la règle "les femmes et les enfants d'abord" qui a guidé les efforts d'évacuation.
""")

st.subheader("Taux de survie par nationalité")
image_survie_nationalite = Image.open("images/12 Taux de survie par nationalité.png")
st.image(image_survie_nationalite, caption="Taux de survie par nationalité", use_column_width=True)
st.write("""
Les passagers d'origine japonaise et chinoise ont présenté un taux de survie élevé, tandis que ceux originaires de pays comme le Danemark et l'Italie ont eu des taux de survie beaucoup plus bas. Ces différences peuvent être attribuées à divers facteurs sociaux et structurels à bord du Titanic.
""")

st.subheader("Distribution de l'âge des survivants par sexe")
image_ages_survie = Image.open("images/10 Distribution de l'âge par sexe et survie.png")
st.image(image_ages_survie, caption="Distribution de l'âge des survivants par sexe", use_column_width=True)
st.write("""
La survie des jeunes passagers, en particulier des enfants, était significativement plus élevée, en particulier pour les filles. Les hommes âgés, quant à eux, avaient les taux de survie les plus bas.
""")

# Section sur les modèles prédictifs
st.header("Modélisation et Prédiction")
st.write("""
Nous avons utilisé plusieurs modèles pour prédire la survie des passagers, notamment la régression logistique, Random Forest et XGBoost. Voici un aperçu des performances de ces modèles, avec les courbes ROC correspondantes.
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
Le modèle Random Forest a identifié les variables les plus influentes dans la prédiction de la survie. Le sexe (en particulier le fait d'être un homme), l'âge et le tarif du billet étaient des facteurs déterminants dans les probabilités de survie.
""")
image_importance = Image.open("images/25 Importance des 10 principales variables selon Random Forest.png")
st.image(image_importance, caption="Importance des variables selon Random Forest", use_column_width=True)

# Section Conclusion
st.header("Conclusion")
st.write("""
L'analyse des données des passagers du Titanic met en lumière des différences frappantes dans les chances de survie selon le sexe, la classe sociale et la nationalité. Les femmes et les enfants, ainsi que les passagers des premières classes, avaient de bien meilleures chances de survie.
""")

st.write("""
Les modèles prédictifs, tels que Random Forest et XGBoost, ont démontré des performances solides, atteignant des scores AUC proches de 0.80. Ces modèles confirment l'importance des variables démographiques et socio-économiques dans la prédiction des chances de survie.
""")

st.write("""
Cette analyse a permis de souligner non seulement des facteurs évidents comme le sexe et la classe, mais aussi des variables moins discutées, telles que la nationalité. L'influence de la langue parlée à bord et des réseaux sociaux à bord a probablement joué un rôle dans les taux de survie.
""")

st.write("""
En conclusion, les données montrent que les inégalités présentes à bord du Titanic ont fortement influencé les chances de survie. Ces résultats mettent en lumière des dynamiques sociales et structurelles encore pertinentes dans les analyses modernes de catastrophes similaires.
""")

st.write("Merci d'avoir consulté ce projet ! Pour plus de détails, n'hésitez pas à explorer le code complet et les analyses supplémentaires sur mon portfolio GitHub.")





