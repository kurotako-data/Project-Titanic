#import pandas as pd
#import pickle
#import numpy as np

import streamlit as st
from PIL import Image


# Configuration de la page
st.set_page_config(page_title="Projet Titanic - Portfolio Data Analyst", layout="wide")

# Menu de navigation dans la barre latérale
st.sidebar.title("Navigation")
page = st.sidebar.radio("Accéder à", ["Introduction", "Situation Initiale", "Survivants", "Modélisation", "Conclusion"])

# Affichage du contenu en fonction de la sélection dans le menu
if page == "Introduction":
    st.title("Titanic, la survie selon le sexe, l'âge, la classe ... et les facteurs langues/nationalité ?")
    st.write("""
    Dans cette analyse, nous explorons en profondeur les facteurs ayant influencé les chances de survie des passagers du Titanic. En plus du sexe, de l'âge et de la classe sociale, nous nous penchons sur une autre variable intrigante : la nationalité, utilisée en l'absence de la variable langue dans le fichier. À travers cette étude, nous tenterons de répondre à la question : la nationalité a-t-elle joué un rôle dans les chances de survie dans le cadre du processus de compréhension des consignes d'évacuation?
    """)
    st.write("""
    Le Titanic transportait une variété de passagers d’origines diverses, et cette diversité se reflète dans les données que nous analysons ici. Cette classification complexe influence notre analyse en plus des facteurs plus classiques comme le sexe et la classe.
    """)
    # Charger l'image du Titanic
    image = Image.open('images/photo_titanic.jpg') 
    st.image(image, caption='Le Titanic', use_column_width=True)

elif page == "Situation Initiale":
    st.header("Situation Initiale des Passagers")
    st.write("""
    Commençons par explorer la situation initiale des passagers à bord du Titanic, en examinant leur sexe, leur âge, leur classe sociale, ainsi que leur nationalité. Ces variables sont cruciales pour comprendre les inégalités qui ont influencé la survie.
    """)
    
    st.subheader("Répartition des passagers par sexe")
    image_sexe = Image.open("images/5 Répartition des passagers par sexe.png")
    st.image(image_sexe, use_column_width=False, width=600)
    st.write("""
    La majorité des passagers étaient des hommes, représentant environ 75 %. Ce déséquilibre entre les sexes soulève la question suivante : comment cela a-t-il influencé les chances de survie, surtout face à la règle "les femmes et les enfants d'abord" ?
    """)

    st.subheader("Répartition des passagers par nationalité")
    image_nationalite = Image.open("images/15 Répartition des passagers par nationalité (Top 10).png")
    st.image(image_nationalite, use_column_width=False, width=600)
    st.write("""
    Les nationalités dominantes étaient principalement l'**Angleterre**, les **États-Unis**, et l'**Irlande**. Cependant, en tenant compte du contexte de l'époque, beaucoup de passagers "britanniques" provenaient de différentes régions de l'Empire Britannique, comme l'**Australie**, le **Canada** et d'autres colonies. En analysant ces données, nous comprenons que ces passagers avaient des cultures et origines variées, ce qui pourrait avoir influencé leur expérience à bord et leurs chances de survie.
    """)

    st.write("""
    Il est important de noter que le Titanic, étant un navire britannique, les instructions d’évacuation ont forcément été communiquées en anglais. À cette époque, la maîtrise de l’anglais n’était pas aussi répandue qu’aujourd’hui, surtout parmi les passagers issus de pays non anglophones. Cela pourrait avoir impacté leur capacité à comprendre rapidement les directives en situation de crise. Les passagers dont la langue maternelle n’était pas l’anglais ont peut-être été désavantagés lors de l’évacuation. Ce facteur, bien que difficile à quantifier, est crucial pour mieux comprendre les disparités de survie selon la nationalité.
    """)

    st.subheader("Distribution des âges des passagers")
    image_ages = Image.open("images/4 Distribution des âges des passagers.png")
    st.image(image_ages, use_column_width=False, width=600)
    st.write("""
    L'âge des passagers était principalement concentré entre 20 et 40 ans. Cette variable, tout comme le sexe, a grandement influencé les chances de survie, les plus jeunes ayant un avantage certain par rapport aux adultes plus âgés.
    """)

elif page == "Survivants":
    st.header("Caractéristiques des Survivants")
    st.write("""
    Nous allons maintenant examiner les caractéristiques des survivants en fonction des mêmes variables : sexe, âge, nationalité et classe. Il est intéressant de voir comment les différences initiales dans la répartition des passagers ont affecté les résultats de la survie.
    """)
    
    st.subheader("Taux de survie par sexe")
    image_survie_sexe = Image.open("images/7 Taux de survie par sexe.png")
    st.image(image_survie_sexe, use_column_width=False, width=600)
    st.write("""
    Les femmes ont eu un taux de survie bien supérieur à celui des hommes, avec environ 75 % des femmes ayant survécu contre seulement 20 % des hommes. Cela souligne l'application de la règle "les femmes et les enfants d'abord" dans un contexte de panique.
    """)

    st.subheader("Taux de survie par nationalité")
    image_survie_nationalite = Image.open("images/12 Taux de survie par nationalité.png")
    st.image(image_survie_nationalite, use_column_width=False, width=800)  # Image agrandie
    st.write("""
    Parmi les nationalités, les passagers issus du **Royaume-Uni**, des **États-Unis** et de l'**Irlande** étaient les plus nombreux. Toutefois, des groupes moins représentés comme les **Syriens** et les **Suédois** ont eu des taux de survie relativement élevés. Ce constat est intéressant car il montre que les groupes minoritaires ont, dans certains cas, bénéficié de meilleures chances de survie.
    Cependant, il faut garder à l'esprit que l'anglais n'était pas une langue universelle à l'époque, et que les passagers dont la langue maternelle n'était pas l'anglais ont potentiellement eu plus de difficulté à comprendre les instructions cruciales lors de l'évacuation.
    """)

    st.subheader("Distribution de l'âge des survivants par sexe")
    image_ages_survie = Image.open("images/10 Distribution de l'âge par sexe et survie.png")
    st.image(image_ages_survie, use_column_width=False, width=600)
    st.write("""
    Les jeunes, en particulier les jeunes filles, ont eu les meilleures chances de survie. Les hommes plus âgés, quant à eux, ont vu leurs chances diminuer, surtout après 50 ans. La combinaison de l'âge et du sexe a donc joué un rôle déterminant dans la survie.
    """)

elif page == "Modélisation":
    st.header("Modélisation et Prédiction")
    st.write("""
    Afin de comprendre comment ces variables ont influencé les chances de survie, nous avons utilisé des modèles prédictifs tels que **Random Forest** et **XGBoost**. Ces modèles nous permettent d'analyser l'importance relative des différentes variables sur les probabilités de survie.
    """)

    # Affichage des résultats des modèles
    st.subheader("Courbes ROC des différents modèles")
    st.write("Voici les courbes ROC pour les modèles que nous avons utilisés pour prédire les chances de survie.")
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

elif page == "Conclusion":
    st.header("Conclusion")
    st.write("""
    Cette analyse approfondie montre que des facteurs tels que le sexe, l'âge, et la classe sociale ont eu une influence importante sur la survie. Cependant, la **nationalité** se révèle également être une variable intrigante, notamment pour les passagers issus des régions périphériques de l'Empire Britannique, comme l'Irlande ou les colonies asiatiques. Le contexte de l'époque, avec une forte présence de passagers britanniques de diverses origines culturelles, a sans doute influencé ces résultats.
    """)
    st.write("""
    En conclusion, l’analyse des données du Titanic révèle des inégalités frappantes dans les chances de survie, mais elle montre également que des facteurs culturels et géographiques, tels que la nationalité, peuvent aussi avoir influencé ces résultats de manière subtile.
    """)
    st.write("Merci d'avoir consulté ce projet ! Pour plus de détails et d'analyses supplémentaires, vous pouvez visiter mon portfolio GitHub.")






