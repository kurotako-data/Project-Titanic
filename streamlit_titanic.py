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
    Les nationalités dominantes étaient principalement l'**Angleterre**, les **États-Unis**, et l'**Irlande**. Cependant, en tenant compte du contexte de l'époque, beaucoup de passagers "britanniques" provenaient de différentes régions de l'Empire Britannique, comme l'**Australie**, le **Canada** et d'autres colonies tout comme de pays n'appartenant pas à l'Empire. En analysant ces données, nous comprenons que ces passagers avaient des cultures et origines variées, ce qui pourrait avoir influencé leur expérience à bord et leurs chances de survie.
    """)

    st.write("""
    Il est important de noter que le Titanic, étant un navire britannique, les instructions d’évacuation ont forcément été communiquées en anglais. À cette époque, la maîtrise de l’anglais n’était pas aussi répandue qu’aujourd’hui, surtout parmi les passagers issus de pays non anglophones. Cela pourrait avoir impacté leur capacité à comprendre rapidement les directives en situation de crise. Les passagers dont la langue maternelle n’était pas l’anglais (Les Suédois ou les Japonais, présent à bord par exemple) ont peut-être été désavantagés lors de l’évacuation. Ce facteur, bien que difficile à quantifier, est crucial pour mieux comprendre les disparités de survie selon la nationalité.
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
    Afin de mieux comprendre quels facteurs ont influencé les chances de survie des passagers du Titanic, nous avons utilisé plusieurs modèles de classification : **Random Forest**, **XGBoost**, et un **Voting Classifier** (combinaison de plusieurs modèles).
    
    Ces modèles nous permettent d’évaluer l’importance des variables telles que le sexe, l’âge, la classe sociale, et la nationalité, et de prédire les probabilités de survie en fonction de ces facteurs. L’objectif est de déterminer quelles caractéristiques ont le plus influencé les chances de survie, et d’évaluer la performance de chaque modèle en fonction de sa capacité à prédire correctement la survie.
    """)

    # Explication des modèles utilisés
    st.subheader("1. Modèles utilisés")
    st.write("""
    - **Random Forest** : Un ensemble d'arbres de décision qui améliore la précision de la prédiction en agrégeant les résultats de plusieurs arbres.
    - **XGBoost** : Un modèle de gradient boosting qui optimise les prédictions en construisant successivement des modèles plus précis, en corrigeant les erreurs des modèles précédents.
    - **Voting Classifier** : Une approche qui combine les prédictions de plusieurs modèles (dans notre cas, Random Forest, XGBoost et un modèle de régression logistique) pour améliorer la robustesse des résultats.
    """)

    # Affichage des courbes ROC
    st.subheader("2. Courbes ROC des modèles")
    st.write("""
    Les courbes ROC nous permettent d'évaluer la performance des modèles de prédiction. Une courbe ROC montre la sensibilité (ou rappel) en fonction du taux de faux positifs. Plus la courbe se rapproche du coin supérieur gauche, meilleure est la performance du modèle. La surface sous la courbe (AUC) est un indicateur clé : plus l’AUC est élevé, meilleur est le modèle.
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        image_roc_rf = Image.open("images/21 courbe ROC  RF.png")
        st.image(image_roc_rf, caption="ROC Random Forest", use_column_width=False, width=400)
        st.write("""
        **AUC** (Random Forest) : 0.86
        Le modèle Random Forest montre de bonnes performances, avec un AUC de 0.86, indiquant qu’il discrimine bien entre les passagers survivants et ceux décédés. Il met particulièrement en avant le sexe (être une femme) comme variable clé pour prédire la survie.
        """)

    with col2:
        image_roc_xgb = Image.open("images/22 courbe ROC  XGB.png")
        st.image(image_roc_xgb, caption="ROC XGBoost", use_column_width=False, width=400)
        st.write("""
        **AUC** (XGBoost) : 0.88
        Le modèle XGBoost surpasse légèrement le Random Forest avec un AUC de 0.88. Ce modèle permet de mieux capter les interactions complexes entre les variables (par exemple, le fait d’être un homme jeune en première classe améliore les chances de survie).
        """)

    with col3:
        image_roc_voting = Image.open("images/23 courbe ROC Voting classifier.png")
        st.image(image_roc_voting, caption="ROC Voting Classifier", use_column_width=False, width=400)
        st.write("""
        **AUC** (Voting Classifier) : 0.89
        Le Voting Classifier, qui combine les résultats de plusieurs modèles, offre la meilleure performance avec un AUC de 0.89. Ce modèle profite des forces de chaque algorithme pour améliorer la robustesse des prédictions.
        """)

    # Analyse de l'importance des variables
    st.subheader("3. Importance des Variables")
    st.write("""
    L’un des grands avantages du modèle **Random Forest** est qu’il permet d’identifier les variables qui ont le plus contribué à la prédiction de la survie. Voici les principales variables influentes :
    
    - **Sexe** : Être une femme augmente significativement les chances de survie, conformément à la règle "les femmes et les enfants d'abord".
    - **Classe sociale** : Les passagers de première classe ont eu de bien meilleures chances de survie que ceux des classes inférieures.
    - **Âge** : Les enfants ont eu de meilleures chances de survie, surtout ceux accompagnés de leurs parents.
    - **Nationalité** : Bien que moins influente que le sexe ou la classe, la nationalité semble avoir joué un rôle pour certains groupes. Par exemple, les passagers des États-Unis et du Royaume-Uni ont eu des taux de survie plus élevés.
    """)

    # Affichage de l'image sur l'importance des variables
    image_importance = Image.open("images/25 Importance des 10 principales variables selon Random Forest.png")
    st.image(image_importance, use_column_width=False, width=600)
    st.write("""
    Le graphique ci-dessus montre que le **sexe**, l’**âge**, et la **classe** sont les trois variables les plus importantes pour prédire la survie. En revanche, la **nationalité** a eu un impact plus limité mais non négligeable.
    """)

    # Résumé de la modélisation
    st.subheader("4. Résumé et Conclusions de la Modélisation")
    st.write("""
    En résumé, nos modèles montrent que les variables liées au **sexe**, à l'**âge**, et à la **classe** ont été les plus déterminantes dans la survie des passagers du Titanic. Le fait d’être une femme en première classe, ou un enfant, a nettement amélioré les chances de survie.
    
    La **nationalité**, bien qu’elle ait un impact plus subtil, a influencé les résultats. Les passagers provenant de pays anglophones, tels que le Royaume-Uni et les États-Unis, semblent avoir mieux compris les consignes d’évacuation, ce qui a pu jouer en leur faveur. Cependant, d’autres facteurs comme la richesse et les connexions sociales ont probablement aussi contribué à cet effet.
    
    Nos modèles, notamment le **Voting Classifier**, offrent des prédictions robustes avec un AUC proche de 0.89, ce qui montre une bonne capacité à discriminer les survivants des autres passagers.
    """)

elif page == "Conclusion":
    st.header("Conclusion")
    st.write("""
    Cette analyse approfondie montre que des facteurs tels que le sexe, l'âge, et la classe sociale ont eu une influence importante sur la survie. Cependant, la **nationalité** se révèle également être une variable intrigante, notamment pour les passagers issus des régions périphériques de l'Empire Britannique, comme l'Irlande ou les colonies asiatiques. Le contexte de l'époque, avec une forte présence de passagers britanniques de diverses origines culturelles, a sans doute influencé ces résultats.
    """)
    st.write("""
    En conclusion, l’analyse des données du Titanic révèle des inégalités frappantes dans les chances de survie, mais elle montre également que des facteurs culturels et géographiques, tels que la nationalité, peuvent aussi avoir influencé ces résultats de manière subtile.
    """)
    st.write("Merci d'avoir consulté ce projet ! Pour plus de détails et d'analyses supplémentaires, vous pouvez visiter mon portfolio GitHub.")






