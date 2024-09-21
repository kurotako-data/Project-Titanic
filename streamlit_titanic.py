#import pandas as pd
#import pickle
#import numpy as np

import streamlit as st
from PIL import Image

# Configuration de la page
st.set_page_config(page_title="Projet Titanic - Portfolio Data Analyst", layout="wide")

# Titre principal
st.title("Titanic, la survie selon le sexe, l'âge, la classe ... et pourquoi pas, la nationalité ?")
st.write("""
Ce projet explore en profondeur les facteurs ayant influencé les chances de survie des passagers du Titanic en se concentrant principalement sur des variables démographiques comme le sexe, l'âge, la classe sociale et, de manière surprenante, la nationalité. Nous allons découvrir comment ces différents éléments ont joué un rôle crucial dans la survie des passagers.
""")

# Section Introduction
st.header("Introduction")
st.write("""
Le naufrage du Titanic est une tragédie qui continue de fasciner. Mais au-delà de l'histoire, les données nous permettent d'explorer en détail ce qui a influencé les chances de survie des passagers.
Ce projet se penche sur les variables clés de survie : le sexe, l'âge, la classe sociale, et une analyse inédite de l'impact potentiel de la nationalité sur les probabilités de survie.
""")
st.write("""
Les résultats nous montrent que les femmes et les enfants, ainsi que les passagers de première classe, avaient de bien meilleures chances de survie. Mais nous allons aussi nous demander si, dans une catastrophe de cette envergure, la nationalité a pu jouer un rôle.
""")

# Charger l'image du Titanic
image = Image.open('images/photo_titanic.jpg') 
st.image(image, caption='Le Titanic', use_column_width=True)

# Section sur la situation initiale des passagers
st.header("Situation Initiale des Passagers")
st.write("""
Examinons la composition des passagers du Titanic avant le naufrage, avec une attention particulière sur le sexe, l'âge, la classe sociale, et la nationalité. Ces facteurs nous aideront à comprendre l'impact qu'ils ont eu sur les probabilités de survie.
""")

st.subheader("Répartition des passagers par sexe")
image_sexe = Image.open("images/5 Répartition des passagers par sexe.png")
st.image(image_sexe, caption="Répartition des passagers par sexe", use_column_width=False, width=600)
st.write("""
Comme attendu, les hommes représentaient environ 75 % des passagers. Cela soulève une question cruciale : comment ce déséquilibre en termes de sexe a-t-il affecté les taux de survie, notamment face à la règle "les femmes et les enfants d'abord" ?
""")

st.subheader("Répartition des passagers par nationalité")
image_nationalite = Image.open("images/15 Répartition des passagers par nationalité (Top 10).png")
st.image(image_nationalite, caption="Répartition des passagers par nationalité (Top 10)", use_column_width=False, width=600)
st.write("""
La majorité des passagers étaient britanniques, suivis de près par des Américains et des Irlandais. Ce point nous amène à nous demander : dans quelle mesure la nationalité a-t-elle influencé les chances de survie ?
""")

st.subheader("Distribution des âges des passagers")
image_ages = Image.open("images/4 Distribution des âges des passagers.png")
st.image(image_ages, caption="Distribution des âges des passagers", use_column_width=False, width=600)
st.write("""
L'âge des passagers était principalement concentré entre 20 et 40 ans. Toutefois, les plus jeunes enfants avaient un avantage sur les adultes en termes de survie, un point que nous explorerons plus en profondeur dans la section des survivants.
""")

# Section sur les survivants
st.header("Caractéristiques des Survivants")
st.write("""
Maintenant que nous avons un aperçu de la situation initiale des passagers, concentrons-nous sur ceux qui ont survécu. Nous allons examiner de plus près les taux de survie en fonction du sexe, de l'âge, de la classe et, encore une fois, de la nationalité.
""")

st.subheader("Taux de survie par sexe")
image_survie_sexe = Image.open("images/7 Taux de survie par sexe.png")
st.image(image_survie_sexe, caption="Taux de survie par sexe", use_column_width=False, width=600)
st.write("""
Sans surprise, les femmes avaient un taux de survie bien supérieur à celui des hommes. Près de 75 % des femmes ont survécu, contre seulement 20 % des hommes. Le facteur du sexe est donc le plus déterminant dans cette tragédie.
""")

st.subheader("Taux de survie par nationalité")
image_survie_nationalite = Image.open("images/12 Taux de survie par nationalité.png")
st.image(image_survie_nationalite, caption="Taux de survie par nationalité", use_column_width=False, width=600)
st.write("""
La nationalité a-t-elle vraiment joué un rôle ? Les données montrent que certains groupes, comme les passagers japonais et chinois, ont eu des taux de survie très élevés, tandis que ceux d'autres pays, comme les Italiens et les Danois, ont eu un taux de survie beaucoup plus faible. 
Bien que ces résultats soient fascinants, il est difficile de conclure s'ils sont liés à des facteurs structurels à bord ou à des circonstances fortuites.
""")

st.subheader("Distribution de l'âge des survivants par sexe")
image_ages_survie = Image.open("images/10 Distribution de l'âge par sexe et survie.png")
st.image(image_ages_survie, caption="Distribution de l'âge des survivants par sexe", use_column_width=False, width=600)
st.write("""
La distribution des âges montre que les jeunes filles avaient les meilleures chances de survie, en particulier celles de moins de 15 ans. Pour les hommes, les chances de survie diminuaient avec l'âge, surtout pour ceux de plus de 50 ans.
""")

# Section sur les modèles prédictifs
st.header("Modélisation et Prédiction")
st.write("""
Pour mieux comprendre les facteurs de survie, nous avons utilisé plusieurs modèles de machine learning, dont Random Forest et XGBoost. Ces modèles permettent d'analyser l'impact relatif des différentes variables sur la survie.
""")

# Affichage des résultats des modèles
st.subheader("Courbes ROC des différents modèles")
st.write("Comparons les performances des modèles prédictifs à travers les courbes ROC.")
col1, col2, col3 = st.columns(3)
with col1:
    image_roc_rf = Image.open("images/21 courbe ROC  RF.png")
    st.image(image_roc_rf, caption="ROC Random Forest", use_column_width=False, width=400)
with col2:
    image_roc_xgb = Image.open("images/22 courbe ROC  XGB.png")
    st.image(image_roc_xgb, caption="ROC XGBoost", use_column_width=False, width=400)
with col3:
    image_roc_voting = Image.open("images/23 courbe ROC Voting classifier.png")
    st.image(image_roc_voting, caption="ROC Voting Classifier", use_column_width=False, width=400)

# Section sur l'importance des variables
st.header("Importance des Variables")
st.write("""
Le modèle Random Forest nous permet d'identifier les variables les plus importantes dans la prédiction de la survie. Le sexe (en particulier être un homme), l'âge et le tarif du billet étaient des facteurs déterminants.
""")
image_importance = Image.open("images/25 Importance des 10 principales variables selon Random Forest.png")
st.image(image_importance, caption="Importance des variables selon Random Forest", use_column_width=False, width=600)

# Section Conclusion
st.header("Conclusion")
st.write("""
Cette analyse met en lumière plusieurs facteurs clés qui ont influencé la survie lors du naufrage du Titanic. Le sexe, l'âge et la classe étaient des déterminants évidents, mais la nationalité pourrait également avoir joué un rôle inattendu. 
Les passagers britanniques et américains, par exemple, avaient des taux de survie relativement élevés, tandis que ceux d'autres nationalités ont semblé être désavantagés.
""")
st.write("""
Cette analyse nous rappelle que, même dans une catastrophe maritime, les inégalités sociales, qu'elles soient liées au sexe, à la classe ou à la nationalité, ont un impact significatif. En utilisant des modèles prédictifs performants comme Random Forest et XGBoost, nous avons pu confirmer que certaines variables démographiques jouaient un rôle majeur dans la prédiction des chances de survie.
""")

st.write("""
En conclusion, cette exploration des données du Titanic montre que, bien que la tragédie ait touché tout le monde, les chances de survie étaient loin d'être égales. Cela soulève d'importantes questions sur l'organisation sociale à bord, et met en lumière des inégalités qui persistent encore aujourd'hui dans les crises.
""")

st.write("Merci d'avoir consulté ce projet ! Pour plus de détails, consultez mon portfolio GitHub pour le code complet et des analyses supplémentaires.")






