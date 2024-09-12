# Project-Titanic

English version and after French version :

Another perspective on the shipwreck

# Project Titanic: Predicting Passenger Survival

## Overview
The 'Titanic Project' is a data analysis initiative focused on predicting the survival of passengers aboard the Titanic based on various key factors. Using a well-known dataset from the Titanic disaster, this project aims to apply machine learning techniques and data analysis to provide insights into the dynamics of survival during this tragic event.

## Objectives
The main objectives of this project are:
1. Predict Survival Based on Nationality: Investigate the influence of passengers' nationality on their likelihood of survival.
2. Explore the Role of Class and Gender: Understand how class (1st, 2nd, 3rd) and gender (male, female) affected survival rates.
3. Develop a Global Predictive Model: Build a predictive model that integrates nationality, class, and gender to forecast the survival of passengers.
4. Additional Insights: Analyze additional variables such as age and fare to enhance the model's accuracy and provide further insights.

## Datasets
We used the famous 'Titanic dataset', which contains details about passengers, their survival status, and other relevant features. The key variables used in this project include:
- Passenger Information: Name, gender, and nationality.
- Class: Class of travel (1st, 2nd, 3rd).
- Survival Status: Whether the passenger survived (1 = yes, 0 = no).
- Age and Fare: Additional features such as the passenger's age and the price of their ticket.

## Data Analysis Approach
The project follows a structured approach:
1. Data Cleaning: Handling missing values, outliers, and standardizing variable formats.
2. Exploratory Data Analysis (EDA): Investigating survival rates based on nationality, class, and gender using visualizations and descriptive statistics.
3. Feature Engineering: Creating new features or refining existing ones to improve model performance.
4. Modeling: Using machine learning models such as **Logistic Regression**, **Random Forest**, and **XGBoost** to predict survival.
5. Evaluation: Evaluating model performance using accuracy, precision, recall, and other metrics. Model comparison to determine the best approach.

## Tools & Technologies
The following tools and technologies were used for this project:
- Python: The main programming language for data analysis and modeling.
- Pandas & NumPy: For data manipulation and cleaning.
- Matplotlib & Seaborn: For data visualization.
- Scikit-learn: For building and evaluating machine learning models.
- Jupyter Notebooks: For documenting and presenting the analysis process.
- GitHub: Version control and project management.

## Results & Insights
By analyzing the data, we found that:
- 'Gender' played a critical role in survival, with females having a significantly higher chance of survival than males.
- 'Class' also impacted survival, with first-class passengers having the highest survival rates, followed by second and third class.
- 'Nationality', although less frequently discussed in Titanic analyses, showed interesting patterns of survival likelihood based on embarkation points and countries of origin.

## Next Steps
Future improvements to this project may include:
1. Incorporating more advanced machine learning models like **XGBoost** and **Gradient Boosting** to improve prediction accuracy.
2. Investigating the impact of more nuanced features like family size, relationships between passengers, and fare per person.
3. Deploying the predictive model using a web interface (e.g., **Streamlit** or **Flask**) to make the model interactive for users.

## Conclusion
This project showcases the use of data analysis and machine learning to explore survival patterns in the Titanic disaster. The combination of nationality, class, and gender provides a unique perspective on survival rates, adding depth to the analysis of this historical event.

## How to Use
To explore this project:
1. Clone the repository:
   ```bash
   git clone https://github.com/ton-utilisateur/Project-Titanic.git

French version :

Une autre perspective sur le naufrage

# Projet Titanic : Prévoir la survie des passagers

## Vue d'ensemble
Le « Projet Titanic » est une initiative d'analyse de données visant à prédire la survie des passagers à bord du Titanic en fonction de divers facteurs clés. En utilisant un ensemble de données bien connu de la catastrophe du Titanic, ce projet vise à appliquer des techniques d'apprentissage automatique et d'analyse de données pour fournir des informations sur la dynamique de la survie au cours de cet événement tragique.

## Objectifs
Les principaux objectifs de ce projet sont les suivants
1. Prédire la survie en fonction de la nationalité : Étudier l'influence de la nationalité des passagers sur leurs chances de survie.
2. Explorer le rôle de la classe et du sexe : Comprendre comment la classe (1ère, 2ème, 3ème) et le sexe (homme, femme) affectent les taux de survie.
3. Développer un modèle prédictif global : Construire un modèle prédictif qui intègre la nationalité, la classe et le sexe pour prévoir la survie des passagers.
4. Perspectives supplémentaires : Analyser des variables supplémentaires telles que l'âge et le tarif pour améliorer la précision du modèle et fournir d'autres informations.

## Ensembles de données
Nous avons utilisé le célèbre « ensemble de données Titanic », qui contient des détails sur les passagers, leur statut de survie et d'autres caractéristiques pertinentes. Les principales variables utilisées dans ce projet sont les suivantes
- Informations sur les passagers : Nom, sexe et nationalité.
- Classe : Classe de voyage (1ère, 2ème, 3ème).
- Statut de survie : Le passager a survécu ou non (1 = oui, 0 = non).
- Âge et tarif : Caractéristiques supplémentaires telles que l'âge du passager et le prix de son billet.

## Approche de l'analyse des données
Le projet suit une approche structurée :
1. Nettoyage des données : Traitement des valeurs manquantes, des valeurs aberrantes et normalisation des formats de variables.
2. Analyse exploratoire des données (AED) : Étude des taux de survie en fonction de la nationalité, de la classe et du sexe à l'aide de visualisations et de statistiques descriptives.
3. Ingénierie des caractéristiques : Création de nouvelles caractéristiques ou amélioration des caractéristiques existantes afin d'améliorer les performances du modèle.
4. Modélisation : Utilisation de modèles d'apprentissage automatique tels que **Régression logistique**, **Forêt aléatoire** et **XGBoost** pour prédire la survie.
5. Évaluation : Évaluation des performances du modèle à l'aide de l'exactitude, de la précision, du rappel et d'autres mesures. Comparaison des modèles pour déterminer la meilleure approche.

## Outils et technologies
Les outils et technologies suivants ont été utilisés pour ce projet :
- Python : Le principal langage de programmation pour l'analyse des données et la modélisation.
- Pandas et NumPy : Pour la manipulation et le nettoyage des données.
- Matplotlib & Seaborn : Pour la visualisation des données.
- Scikit-learn : Pour la construction et l'évaluation de modèles d'apprentissage automatique.
- Jupyter Notebooks : Pour documenter et présenter le processus d'analyse.
- GitHub : Contrôle des versions et gestion de projet.

## Résultats et perspectives
L'analyse des données nous a permis de constater que :
- Le « sexe » joue un rôle essentiel dans la survie, les femmes ayant des chances de survie nettement plus élevées que les hommes.
- La « classe » a également eu un impact sur la survie, les passagers de première classe ayant les taux de survie les plus élevés, suivis par ceux de deuxième et troisième classe.
- La nationalité, bien que moins souvent abordée dans les analyses du Titanic, a montré des schémas intéressants de probabilité de survie en fonction des points d'embarquement et des pays d'origine.

## Prochaines étapes
Les améliorations futures de ce projet pourraient inclure
1. Incorporer des modèles d'apprentissage automatique plus avancés tels que **XGBoost** et **Gradient Boosting** afin d'améliorer la précision des prédictions.
2. Étudier l'impact de caractéristiques plus nuancées telles que la taille de la famille, les relations entre les passagers et le tarif par personne.
3. Déployer le modèle prédictif à l'aide d'une interface web (par exemple **Streamlit** ou **Flask**) pour rendre le modèle interactif pour les utilisateurs.

## Conclusion
Ce projet illustre l'utilisation de l'analyse de données et de l'apprentissage automatique pour explorer les modèles de survie dans la catastrophe du Titanic. La combinaison de la nationalité, de la classe et du sexe fournit une perspective unique sur les taux de survie, ajoutant de la profondeur à l'analyse de cet événement historique.

## Comment utiliser
Pour explorer ce projet :
1. Cloner le dépôt :
    ```bash
   git clone https://github.com/ton-utilisateur/Project-Titanic.git
