#projet_titanic_python


# Importation des bibliothèques nécessaires
import pandas as pd
import numpy as np
# Importer les bibliothèques de visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# Charger le fichier Titanic.csv
file_path = 'data/titanic.csv'  
titanic_data = pd.read_csv(file_path, sep=';')

# Afficher les informations du dataset pour détecter les valeurs manquantes
missing_values = titanic_data.isnull().sum()
print(missing_values)

# Afficher les types de données pour vérifier si certaines colonnes nécessitent un formatage
data_types = titanic_data.dtypes
print(data_types)


# Suppression de la colonne "ticketno" (car elle est entièrement manquante)
titanic_data.drop(columns=['ticketno'], inplace=True)

#Les colonnes Nbr des frères et sœurs / conjoints à bord du Titanic et Nbr des parents / enfants à bord du Titanic ont 900 valeurs manquantes, 
#ce qui est important. Ces colonnes ne sont pas essentielles pour notre analyse. 
titanic_data.drop(columns=['Nbr des frères et sœurs / conjoints à bord du Titanic', 
                           'Nbr des parents / enfants à bord du Titanic'], inplace=True)

# Convertir la colonne 'survived' en valeurs numériques
titanic_data['survived'] = titanic_data['survived'].map({'yes': 1, 'no': 0})

# Imputation des valeurs manquantes

# 1. Imputer l'âge avec la médiane
titanic_data = titanic_data.assign(age=titanic_data['age'].fillna(titanic_data['age'].median()))

# 2. Imputer le prix du billet (fare) avec la moyenne
titanic_data = titanic_data.assign(fare=titanic_data['fare'].fillna(titanic_data['fare'].mean()))

# 3. Imputer la nationalité (country) avec "Unknown"
titanic_data = titanic_data.assign(country=titanic_data['country'].fillna('Unknown'))

# Vérification des valeurs manquantes après imputation
print(titanic_data.isnull().sum())

# Affichage des premières lignes pour vérifier les changements
print(titanic_data.head())


# Configurer le style de visualisation
#sns.set(style="whitegrid")

# 1. Histogramme de la distribution des âges
plt.figure(figsize=(10, 6))
sns.histplot(titanic_data['age'], bins=30, kde=False, color='blue')
plt.title('Distribution des âges des passagers')
plt.xlabel('Âge')
plt.ylabel('Nombre de passagers')
plt.show()

# 2. Diagramme en barres des sexes avec `hue`
plt.figure(figsize=(6, 4))
sns.countplot(data=titanic_data, x='gender', hue='gender', palette='Set2', legend=False)
plt.title('Répartition des passagers par sexe')
plt.xlabel('Sexe')
plt.ylabel('Nombre de passagers')
plt.show()


# 3. Diagramme en barres de la répartition par classe avec `hue`
plt.figure(figsize=(12, 4))
sns.countplot(data=titanic_data, x='class', hue='class', palette='Set3', legend=False)
plt.title('Répartition des passagers par classe')
plt.xlabel('Classe')
plt.ylabel('Nombre de passagers')
plt.show()

# 4. Comparaison des taux de survie par sexe avec `hue` et palette
plt.figure(figsize=(6, 4))
sns.barplot(data=titanic_data, x='gender', y='survived', hue='gender', palette='Set1', dodge=False, legend=False)
plt.title('Taux de survie par sexe')
plt.xlabel('Sexe')
plt.ylabel('Taux de survie')
plt.show()


# 5. Comparaison des taux de survie par classe avec `hue` et palette
plt.figure(figsize=(12, 4))
sns.barplot(data=titanic_data, x='class', y='survived', hue='class', palette='Set1', dodge=False, legend=False)
plt.title('Taux de survie par classe')
plt.xlabel('Classe')
plt.ylabel('Taux de survie')
plt.show()

# 6. Boxplot de l'âge par classe et survie
plt.figure(figsize=(12, 6))
sns.boxplot(data=titanic_data, x='class', y='age', hue='survived', palette='Set2')
plt.title('Distribution de l\'âge par classe et survie')
plt.xlabel('Classe')
plt.ylabel('Âge')
plt.show()

# 7. Boxplot de l'âge par sexe et survie
plt.figure(figsize=(10, 6))
sns.boxplot(data=titanic_data, x='gender', y='age', hue='survived', palette='Set3')
plt.title('Distribution de l\'âge par sexe et survie')
plt.xlabel('Sexe')
plt.ylabel('Âge')
plt.show()

# Examiner la corrélation entre les variables numériques pour voir lesquelles pourraient avoir une relation avec la survie.
# Cela peut être utile pour la modélisation.

# Sélectionner uniquement les colonnes numériques
numeric_columns = titanic_data.select_dtypes(include=['float64', 'int64'])

# Matrice de corrélation entre les variables numériques
correlation_matrix = numeric_columns.corr()

# Visualisation de la matrice de corrélation avec une heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title('Matrice de corrélation des variables numériques')
plt.show()

# Analyse des variables catégorielles avec des variables comme le sexe, la classe, et la nationalité. 
# afin de regarder comment ces variables affectent la survie de manière plus approfondie.
# Code pour une heatmap croisée entre ces variables catégorielles et la survie :

# Tableau croisé dynamique pour explorer la relation entre classe, sexe et survie
pivot_table = pd.pivot_table(titanic_data, values='survived', index=['class'], columns=['gender'], aggfunc='mean')

# Visualisation avec une heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(pivot_table, annot=True, cmap="Blues", linewidths=0.5)
plt.title('Taux de survie par sexe et classe')
plt.show()

#Hypothèses avancées pour la modélisation basées sur les visualisations faites :

# Les femmes ont un taux de survie plus élevé que les hommes (surtout en première et deuxième classes).
# Hypothèse : Le sexe est un facteur important de la survie.

# Les passagers de première classe ont un taux de survie plus élevé que ceux des deuxième et troisième classes.
# Hypothèse : La classe est corrélée positivement avec la survie.

# Les jeunes passagers, en particulier les enfants, ont un taux de survie plus élevé.
# Hypothèse : L'âge joue un rôle dans la probabilité de survie.

# Code pour explorer ces relations entre âge, classe, sexe et survie :

# Boxplot de l'âge par classe et survie
plt.figure(figsize=(12, 6))
sns.boxplot(data=titanic_data, x='class', y='age', hue='survived', palette='Set2')
plt.title('Distribution de l\'âge par classe et survie')
plt.xlabel('Classe')
plt.ylabel('Âge')
plt.show()

# Boxplot de l'âge par sexe et survie
plt.figure(figsize=(10, 6))
sns.boxplot(data=titanic_data, x='gender', y='age', hue='survived', palette='Set3')
plt.title('Distribution de l\'âge par sexe et survie')
plt.xlabel('Sexe')
plt.ylabel('Âge')
plt.show()

#Première étape : Préparation des données
#Voici ce que nous allons faire pour cette phase de préparation des données :

#Standardisation des données numériques : Certaines colonnes, comme l'âge et le prix du billet (fare), 
#peuvent avoir des échelles différentes. Nous allons les normaliser afin que toutes les données #
#numériques aient une échelle comparable. Cela est particulièrement utile pour certains modèles comme KNN ou SVM.

#Transformation des variables catégorielles :

#Variables comme le genre et la classe doivent être encodées en variables numériques via one-hot encoding.
#La nationalité (variable country) doit également être encodée afin que l'on puisse inclure 
#cette dimension dans nos modèles.

#Gestion des variables redondantes ou non pertinentes :

#Nous avons déjà supprimé certaines colonnes non pertinentes comme ticketno et d'autres variables ayant 
#beaucoup de valeurs manquantes.
#Nous allons vérifier que toutes les autres variables sont prêtes pour la modélisation.

# Importer les bibliothèques nécessaires
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

# 1. Standardisation des données numériques (âge, prix du billet)
scaler = StandardScaler()
titanic_data[['age', 'fare']] = scaler.fit_transform(titanic_data[['age', 'fare']])

# 2. Transformation des variables catégorielles (one-hot encoding)
titanic_data = pd.get_dummies(titanic_data, columns=['gender', 'class', 'embarked', 'country'], drop_first=True)

# 3. Vérification des données après transformation
print(titanic_data.head())

# Séparation des caractéristiques (features) et de la variable cible (target)
X = titanic_data.drop(columns=['survived'])  # Features (variables explicatives)
y = titanic_data['survived']  # Cible (variable à prédire)

# Vérification de la forme des données
print(X.shape, y.shape)

# 4. Séparation des données d'entraînement et de test (80% entraînement, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vérification de la séparation
print(f'Taille du set d\'entraînement : {X_train.shape[0]} exemples')
print(f'Taille du set de test : {X_test.shape[0]} exemples')

#Étape 2 : Création et entraînement des modèles de machine learning
#Pour cette étape, voici les actions prévues :

#Modèles à tester :

#Régression logistique
#Forêt aléatoire (Random Forest)
#Support Vector Machine (SVM)
#K-Nearest Neighbors (KNN)

#Évaluation des performances :

#Accuracy (précision) : La proportion des prédictions correctes.
#Matrice de confusion : Pour voir les prédictions correctes et incorrectes par classe.
#F1-Score : Une métrique plus équilibrée entre précision et rappel, utile si les classes sont déséquilibrées.
#Validation croisée : Nous allons également utiliser la validation croisée pour obtenir une meilleure estimation des performances du modèle.

# Importer les bibliothèques nécessaires
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler


# Fonction pour évaluer les modèles
def evaluate_model(model, X_train, y_train, X_test, y_test):
    # Entraînement du modèle
    model.fit(X_train, y_train)
    
    # Prédictions
    y_pred = model.predict(X_test)
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Matrice de confusion
    confusion = confusion_matrix(y_test, y_pred)
    
    # Rapport de classification
    report = classification_report(y_test, y_pred)
    
    return accuracy, confusion, report

# Étape 1 : Prétraitement des données
# Séparer les features et la cible
# Suppression de la colonne 'Num' qui est non pertinente
X = titanic_data.drop(columns=['Num', 'survived'])
y = titanic_data['survived']

# Diviser les données en jeu d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Appliquer SMOTE pour rééquilibrer les classes
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Standardisation des données (centrer et réduire)
scaler = StandardScaler()
X_resampled_scaled = scaler.fit_transform(X_resampled)
X_test_scaled = scaler.transform(X_test)

# Étape 2 : Création et évaluation des modèles de machine learning

# Importer les bibliothèques nécessaires
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Rééchantillonnage avec SMOTE pour rééquilibrer les classes
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Standardisation des données (centrer et réduire)
scaler = StandardScaler()
X_resampled_scaled = scaler.fit_transform(X_resampled)
X_test_scaled = scaler.transform(X_test)

# Fonction pour évaluer les modèles
def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Matrice de confusion
    confusion = confusion_matrix(y_test, y_pred)

    # Rapport de classification
    report = classification_report(y_test, y_pred)

    return accuracy, confusion, report

# Grid Search pour la Régression Logistique
logreg = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
param_grid_logreg = {
    'C': [0.01, 0.1, 1, 10, 100],  # Paramètre de régularisation
    'penalty': ['l2'],  # Régularisation l2
}
grid_search_logreg = GridSearchCV(logreg, param_grid_logreg, cv=5, scoring='accuracy')
grid_search_logreg.fit(X_resampled_scaled, y_resampled)

# Meilleurs hyperparamètres et évaluation
best_logreg = grid_search_logreg.best_estimator_
print(f"Meilleurs paramètres Régression Logistique: {grid_search_logreg.best_params_}")

accuracy, confusion, report = evaluate_model(best_logreg, X_resampled_scaled, y_resampled, X_test_scaled, y_test)
print("\nModèle optimisé: Régression Logistique")
print(f"Accuracy: {accuracy:.4f}")
print("Matrice de confusion:")
print(confusion)
print("Rapport de classification:")
print(report)


# Grid Search pour Random Forest
rf = RandomForestClassifier(random_state=42)
param_grid_rf = {
    'n_estimators': [100, 200, 500],  # Nombre d'arbres
    'max_depth': [None, 10, 20, 30],  # Profondeur maximale
    'min_samples_split': [2, 5, 10],  # Minimum d'échantillons pour diviser un noeud
    'min_samples_leaf': [1, 2, 4],    # Minimum d'échantillons dans chaque feuille
}
grid_search_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring='accuracy')
grid_search_rf.fit(X_resampled, y_resampled)

# Meilleurs hyperparamètres et évaluation
best_rf = grid_search_rf.best_estimator_
print(f"\nMeilleurs paramètres Random Forest: {grid_search_rf.best_params_}")

accuracy, confusion, report = evaluate_model(best_rf, X_resampled, y_resampled, X_test, y_test)
print("\nModèle optimisé: Random Forest")
print(f"Accuracy: {accuracy:.4f}")
print("Matrice de confusion:")
print(confusion)
print("Rapport de classification:")
print(report)

#Analyse des résultats après optimisation des hyperparamètres

#1. Régression Logistique Optimisée
#Meilleurs paramètres :
#C = 0.1
#penalty = l2 (régularisation L2)
#Performance :
#Accuracy : 0.7715
#Matrice de confusion :
#253 passagers correctement classés comme non-survivants (classe 0).
#88 passagers correctement classés comme survivants (classe 1).
#50 passagers survivants mal classés comme non-survivants.
#Précision (classe 1) : 0.63 (63% des survivants prédits sont effectivement des survivants).
#Rappel (classe 1) : 0.64 (64% des survivants sont correctement identifiés).
#f1-score (classe 1) : 0.64 (compromis entre précision et rappel).

#Interprétation :

#La régression logistique optimisée offre une meilleure gestion de la régularisation, avec 
#des performances globales raisonnables.
#L'accuracy de 77.15% est stable, mais la précision et le rappel sur les survivants sont encore 
#faibles (autour de 0.63 et 0.64), indiquant que le modèle a du mal à bien détecter les survivants.
#Toutefois, la performance générale reste équilibrée et adéquate compte tenu des limitations des données.

#2. Random Forest Optimisée
#Meilleurs paramètres :
#max_depth = 30
#min_samples_leaf = 1
#min_samples_split = 5
#n_estimators = 500

#Performance :
#Accuracy : 0.7602
#Matrice de confusion :
#256 passagers correctement classés comme non-survivants (classe 0).
#80 passagers correctement classés comme survivants (classe 1).
#58 passagers survivants mal classés comme non-survivants.
#Précision (classe 1) : 0.62
#Rappel (classe 1) : 0.58
#f1-score (classe 1) : 0.60

#Interprétation :

#Random Forest avec ses paramètres optimisés est légèrement moins performante que la régression logistique 
#en termes d'accuracy (76% contre 77.15%).
#Bien que la précision et le rappel pour les survivants (classe 1) restent proches (0.62 et 0.58), 
#cela montre que le modèle a du mal à distinguer correctement les survivants.
#La Random Forest gère mieux les passagers non-survivants, mais a des difficultés similaires à la 
#régression logistique pour identifier les survivants.

#Conclusion
#Régression Logistique optimisée semble légèrement mieux performante que la Random Forest sur cet 
#ensemble de données, avec un léger avantage en termes de précision et de rappel sur les survivants.
#Malgré l'optimisation, les deux modèles présentent des difficultés à identifier correctement les 
#survivants, ce qui est un défi commun avec des classes déséquilibrées comme c'est le cas ici.

#Prochaine étape :
#Rééquilibrer davantage les classes via d'autres techniques comme l'ajustement des poids de classe 
#(ex: dans la régression logistique avec class_weight='balanced') ou l'essai d'autres méthodes 
#d'oversampling ou undersampling.
#Tester d'autres algorithmes comme Gradient Boosting ou XGBoost qui sont plus puissants pour les 
#données complexes et pourraient mieux capter les relations entre variables.

#En fonction de ces observations, nous pourrions envisager des ajustements supplémentaires, 
#ou explorer d'autres algorithmes comme mentionné.

# Ajout du modèle Gradient Boosting 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc, RocCurveDisplay

# Initialisation du modèle Gradient Boosting
gbc = GradientBoostingClassifier(random_state=42)

# Entraînement du modèle sur les données rééchantillonnées et normalisées
gbc.fit(X_resampled_scaled, y_resampled)

# Prédiction sur l'ensemble de test
y_pred_gbc = gbc.predict(X_test_scaled)

# Évaluation du modèle Gradient Boosting
accuracy_gbc = accuracy_score(y_test, y_pred_gbc)
confusion_gbc = confusion_matrix(y_test, y_pred_gbc)
report_gbc = classification_report(y_test, y_pred_gbc)

print("\nModèle: Gradient Boosting")
print(f"Accuracy: {accuracy_gbc:.4f}")
print("Matrice de confusion:")
print(confusion_gbc)
print("Rapport de classification:")
print(report_gbc)

# Visualisation de la courbe ROC pour Gradient Boosting
y_pred_proba_gbc = gbc.predict_proba(X_test_scaled)[:, 1]  # Probabilité pour la classe positive
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_gbc)
roc_auc = auc(fpr, tpr)

# Affichage de la courbe ROC
plt.figure(figsize=(8, 6))
RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='Gradient Boosting').plot()
plt.title('Courbe ROC - Gradient Boosting')
plt.show()

print(f"AUC du modèle Gradient Boosting: {roc_auc:.4f}")

# Comparaison des courbes ROC pour plusieurs modèles
plt.figure(figsize=(10, 8))

# Courbe ROC pour la régression logistique
y_pred_proba_logreg = best_logreg.predict_proba(X_test_scaled)[:, 1]  # Probabilité pour la classe positive
fpr_logreg, tpr_logreg, _ = roc_curve(y_test, y_pred_proba_logreg)
roc_auc_logreg = auc(fpr_logreg, tpr_logreg)
RocCurveDisplay(fpr=fpr_logreg, tpr=tpr_logreg, roc_auc=roc_auc_logreg, estimator_name='Régression Logistique').plot()

# Courbe ROC pour Random Forest
y_pred_proba_rf = best_rf.predict_proba(X_test)[:, 1]  # Probabilité pour la classe positive
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)
RocCurveDisplay(fpr=fpr_rf, tpr=tpr_rf, roc_auc=roc_auc_rf, estimator_name='Random Forest').plot()

# Courbe ROC pour Gradient Boosting
y_pred_proba_gbc = gbc.predict_proba(X_test_scaled)[:, 1]  # Probabilité pour la classe positive
fpr_gbc, tpr_gbc, _ = roc_curve(y_test, y_pred_proba_gbc)
roc_auc_gbc = auc(fpr_gbc, tpr_gbc)
RocCurveDisplay(fpr=fpr_gbc, tpr=tpr_gbc, roc_auc=roc_auc_gbc, estimator_name='Gradient Boosting').plot()

# Afficher le graphique final
plt.title('Comparaison des courbes ROC pour différents modèles')
plt.show()

#Les résultats montrent que le modèle de Gradient Boosting est l'un des plus performants dans cette phase, 
#avec un AUC de 0.81, ce qui en fait le modèle qui discrimine le mieux entre les 
#classes (survivants et non-survivants). 
#Comparé à la régression logistique et à la random forest, qui ont respectivement des 
#AUC de 0.79 et 0.78, il présente un léger avantage.

#Voici les points clés de ces résultats :

#1. Gradient Boosting
#Accuracy : 0.7805
#Précision pour les survivants (classe 1) : 0.67
#Rappel pour les survivants : 0.58
#AUC : 0.81
#Le Gradient Boosting est bien équilibré, et avec son AUC de 0.81, 
#il surpasse légèrement les autres modèles. 
#Cependant, il est toujours un peu en difficulté pour bien prédire les survivants (classe 1), 
#comme en témoigne le rappel à 0.58.

#2. Régression Logistique
#Accuracy : 0.7715
#AUC : 0.79
#C'est un modèle assez fiable avec une bonne régularisation après optimisation, 
#mais il reste moins performant que le Gradient Boosting en termes de détection des survivants.

#3. Random Forest
#Accuracy : 0.7602
#AUC : 0.78
#La random forest est un peu en dessous des autres modèles en termes d'AUC, 
#bien qu'elle soit robuste et performante globalement, en particulier pour prédire les non-survivants.

#Analyse finale :
#Le Gradient Boosting semble offrir un léger avantage par rapport aux autres modèles, 
#principalement en raison de sa meilleure capacité à équilibrer entre précision et rappel 
#pour la classe des survivants. Cependant, il est toujours possible d'améliorer la détection 
#des survivants en ajustant encore les hyperparamètres ou en essayant des approches d'échantillonnage 
#supplémentaires, comme l'ajustement des poids de classe.

#Nous pourrions également examiner les contributions de chaque variable pour voir si certaines 
#peuvent être ajustées ou si d'autres transformations sur les données pourraient 
#améliorer la performance des modèles.


