#projet_titanic_python


# Importation des bibliothèques nécessaires
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Charger le fichier Titanic.csv
file_path = '/Users/Patrice/Documents/GitHub/Project-Titanic/data/titanic.csv' 
titanic_data = pd.read_csv(file_path, sep=';')

# Afficher les informations du dataset pour détecter les valeurs manquantes
missing_values = titanic_data.isnull().sum()
print(missing_values)

# Afficher les types de données pour vérifier si certaines colonnes nécessitent un formatage
data_types = titanic_data.dtypes
print(data_types)

#Les colonnes Nbr des frères et sœurs / conjoints à bord du Titanic et Nbr des parents / enfants à bord du Titanic ont 900 valeurs manquantes, 
#ce qui est important.Tout comme la colonne ticketno. Ces colonnes ne sont pas essentielles pour notre analyse. 

titanic_data.drop(columns=['ticketno', 'Nbr des frères et sœurs / conjoints à bord du Titanic', 'Nbr des parents / enfants à bord du Titanic'], inplace=True)

# Convertir la colonne 'survived' en valeurs numériques
titanic_data['survived'] = titanic_data['survived'].map({'yes': 1, 'no': 0})

# Imputation des valeurs manquantes
titanic_data['age'].fillna(titanic_data['age'].median(), inplace=True)
titanic_data['fare'].fillna(titanic_data['fare'].mean(), inplace=True)
titanic_data['country'].fillna('Unknown', inplace=True)

# Vérification des valeurs manquantes après imputation
print(titanic_data.isnull().sum())

# Affichage des premières lignes pour vérifier les changements
print(titanic_data.head())

# Analyse exploratoire des données (EDA)

# 1. Histogramme de la distribution des âges
plt.figure(figsize=(10, 6))
sns.histplot(titanic_data['age'], bins=30, kde=False, color='blue')
plt.title('Distribution des âges des passagers')
plt.xlabel('Âge')
plt.ylabel('Nombre de passagers')
plt.show()

# 2. Diagramme en barres des sexes
plt.figure(figsize=(6, 4))
sns.countplot(data=titanic_data, x='gender', hue='gender', palette='Set2')
plt.title('Répartition des passagers par sexe')
plt.xlabel('Sexe')
plt.ylabel('Nombre de passagers')
plt.show()

# 3. Diagramme en barres de la répartition par classe
plt.figure(figsize=(12, 4))
sns.countplot(data=titanic_data, x='class', hue='class', palette='Set3')
plt.title('Répartition des passagers par classe')
plt.xlabel('Classe')
plt.ylabel('Nombre de passagers')
plt.show()

# 4. Comparaison des taux de survie par sexe
plt.figure(figsize=(6, 4))
sns.barplot(data=titanic_data, x='gender', y='survived', hue='gender', palette='Set1', dodge=False)
plt.title('Taux de survie par sexe')
plt.xlabel('Sexe')
plt.ylabel('Taux de survie')
plt.show()

# 5. Comparaison des taux de survie par classe
plt.figure(figsize=(12, 4))
sns.barplot(data=titanic_data, x='class', y='survived', hue='class', palette='Set1', dodge=False)
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

# Examiner la corrélation entre les variables numériques
numeric_columns = titanic_data.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numeric_columns.corr()

# Visualisation de la matrice de corrélation avec une heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title('Matrice de corrélation des variables numériques')
plt.show()

# 8. Analyse de la nationalité
# Comparaison des taux de survie par nationalité
plt.figure(figsize=(12, 8))
survived_by_country = titanic_data.groupby('country')['survived'].mean().sort_values(ascending=False)
sns.barplot(x=survived_by_country.index, y=survived_by_country.values, hue=survived_by_country.index, palette='coolwarm')
plt.xticks(rotation=90)
plt.title('Taux de survie par nationalité')
plt.xlabel('Nationalité')
plt.ylabel('Taux de survie')
plt.show()

# 9. Analyse croisée entre classe, sexe et survie avec heatmap
pivot_table = pd.pivot_table(titanic_data, values='survived', index=['class'], columns=['gender'], aggfunc='mean')

plt.figure(figsize=(8, 6))
sns.heatmap(pivot_table, annot=True, cmap="Blues", linewidths=0.5)
plt.title('Taux de survie par sexe et classe')
plt.show()

# 10. Heatmap pour visualiser le taux de survie par nationalité et classe (nationalités avec au moins 10 passagers)
country_class_data = titanic_data.groupby(['country', 'class'])['survived'].mean().unstack()
filtered_countries = titanic_data['country'].value_counts()[titanic_data['country'].value_counts() >= 10].index
filtered_heatmap_data = country_class_data.loc[filtered_countries]

plt.figure(figsize=(10, 6))
sns.heatmap(filtered_heatmap_data, annot=True, cmap="Reds", linewidths=0.5)
plt.title('Taux de survie par nationalité et classe (Top nationalités)')
plt.xticks(rotation=45)
plt.show()

# 11. Diagramme en barres pour la répartition des 10 nationalités les plus fréquentes
top_countries = titanic_data['country'].value_counts().nlargest(10).index  # Les 10 nationalités les plus représentées
filtered_data = titanic_data[titanic_data['country'].isin(top_countries)]

plt.figure(figsize=(12, 6))
sns.countplot(data=filtered_data, x='country', hue='country', palette='Set2', order=top_countries)
plt.title('Répartition des passagers par nationalité (Top 10)')
plt.xlabel('Nationalité')
plt.ylabel('Nombre de passagers')
plt.xticks(rotation=45)
plt.show()

# 12. Heatmap pour visualiser le taux de survie par nationalité et sexe (nationalités avec au moins 10 passagers)
pivot_table_country_gender = titanic_data.pivot_table(values='survived', index='country', columns='gender', aggfunc='mean')
filtered_country_gender = pivot_table_country_gender.loc[filtered_countries]

plt.figure(figsize=(10, 6))
sns.heatmap(filtered_country_gender, annot=True, cmap='Blues', linewidths=0.5)
plt.title('Taux de survie par nationalité et sexe (Top nationalités)')
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
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from imblearn.over_sampling import SMOTE
from sklearn.metrics import RocCurveDisplay
from sklearn.neighbors import KNeighborsClassifier

# Préparation des données avec encodage one-hot
# Encodage one-hot des variables catégorielles
titanic_data_encoded = pd.get_dummies(titanic_data, columns=['gender', 'class', 'embarked', 'country'], drop_first=True)

# Séparation des caractéristiques (features) et de la variable cible (target)
X = titanic_data_encoded.drop(columns=['Num', 'survived'])  # Features (variables explicatives)
y = titanic_data_encoded['survived']  # Cible (variable à prédire)

# Standardisation des données numériques après encodage one-hot
scaler = StandardScaler()
X_resampled_scaled = scaler.fit_transform(X_resampled)
X_test_scaled = scaler.transform(X_test)

# 1. Régression Logistique avec ajustement des poids de classe
logreg = LogisticRegression(solver='lbfgs', max_iter=1000, class_weight='balanced', random_state=42)
param_grid_logreg = {'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l2']}
random_search_logreg = RandomizedSearchCV(logreg, param_distributions=param_grid_logreg, cv=5, scoring='accuracy', n_iter=5, random_state=42)
random_search_logreg.fit(X_resampled_scaled, y_resampled)
best_logreg = random_search_logreg.best_estimator_

# 2. Forêt Aléatoire avec RandomizedSearchCV et ajustement des poids de classe
rf = RandomForestClassifier(class_weight='balanced', random_state=42)
param_grid_rf = {'n_estimators': [100, 200, 500], 'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
random_search_rf = RandomizedSearchCV(rf, param_distributions=param_grid_rf, cv=5, scoring='accuracy', n_iter=10, random_state=42)
random_search_rf.fit(X_resampled_scaled, y_resampled)
best_rf = random_search_rf.best_estimator_

# 3. XGBoost avec RandomizedSearchCV
xgb = XGBClassifier(random_state=42)
param_grid_xgb = {'n_estimators': [100, 200, 500], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7]}
random_search_xgb = RandomizedSearchCV(xgb, param_distributions=param_grid_xgb, cv=5, scoring='accuracy', n_iter=10, random_state=42)
random_search_xgb.fit(X_resampled_scaled, y_resampled)
best_xgb = random_search_xgb.best_estimator_

# 4. K-Nearest Neighbors (KNN)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_resampled_scaled, y_resampled)

# 5. Ensemble learning - Voting Classifier (combinaison de modèles)
ensemble_model = VotingClassifier(estimators=[
    ('logreg', best_logreg), 
    ('rf', best_rf), 
    ('xgb', best_xgb), 
    ('knn', knn)
], voting='soft')  # Le 'soft' utilise les probabilités prédites
ensemble_model.fit(X_resampled_scaled, y_resampled)

# 6. Validation croisée avec StratifiedKFold
skf = StratifiedKFold(n_splits=5)

# Validation croisée pour la régression logistique
cross_val_scores_logreg = cross_val_score(best_logreg, X_resampled_scaled, y_resampled, cv=skf)
print(f'Logistic Regression CV accuracy: {np.mean(cross_val_scores_logreg):.4f}')

# Validation croisée pour la Forêt Aléatoire
cross_val_scores_rf = cross_val_score(best_rf, X_resampled_scaled, y_resampled, cv=skf)
print(f'Random Forest CV accuracy: {np.mean(cross_val_scores_rf):.4f}')

# Validation croisée pour XGBoost
cross_val_scores_xgb = cross_val_score(best_xgb, X_resampled_scaled, y_resampled, cv=skf)
print(f'XGBoost CV accuracy: {np.mean(cross_val_scores_xgb):.4f}')

# Validation croisée pour le Voting Classifier
cross_val_scores_ensemble = cross_val_score(ensemble_model, X_resampled_scaled, y_resampled, cv=skf)
print(f'Voting Classifier CV accuracy: {np.mean(cross_val_scores_ensemble):.4f}')

# 7. Évaluation sur le jeu de test pour tous les modèles
models = {
    'Logistic Regression': best_logreg,
    'Random Forest': best_rf,
    'XGBoost': best_xgb,
    'Voting Classifier': ensemble_model
}

for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'\nModèle: {name}')
    print(f'Accuracy: {accuracy:.4f}')
    print('Matrice de confusion:')
    print(confusion_matrix(y_test, y_pred))
    print('Rapport de classification:')
    print(classification_report(y_test, y_pred))

# 8. Courbes ROC et comparaison des AUC
plt.figure(figsize=(10, 8))

for name, model in models.items():
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]  # Probabilité pour la classe positive
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=name).plot()

plt.title('Comparaison des courbes ROC pour différents modèles')
plt.show()

# Insights complémentaires

# Entraînement de Random Forest avec RandomizedSearchCV
rf = RandomForestClassifier(random_state=42)
param_grid_rf = {
    'n_estimators': [100, 200, 500],  # Nombre d'arbres
    'max_depth': [None, 10, 20, 30],  # Profondeur maximale
    'min_samples_split': [2, 5, 10],  # Minimum d'échantillons pour diviser un noeud
    'min_samples_leaf': [1, 2, 4]     # Minimum d'échantillons dans chaque feuille
}

# Utilisation de RandomizedSearchCV pour trouver les meilleurs hyperparamètres
random_search_rf = RandomizedSearchCV(rf, param_distributions=param_grid_rf, cv=5, scoring='accuracy', random_state=42)
random_search_rf.fit(X_resampled_scaled, y_resampled)

# Stocker le meilleur modèle trouvé
best_rf = random_search_rf.best_estimator_

# Vérifier les paramètres optimaux
print(f'Meilleurs paramètres Random Forest: {random_search_rf.best_params_}')

# Une fois best_rf bien défini, on peut extraire l'importance des variables
importances = best_rf.feature_importances_
feature_names = X.columns
feature_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)

# Affichage des 10 variables les plus importantes
print(feature_importances.head(10))

# Visualisation
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances[:10], y=feature_importances.index[:10])
plt.title('Importance des 10 principales variables selon Random Forest')
plt.xlabel('Importance')
plt.ylabel('Variables')
plt.show()

# Test de Chi-carré pour l'indépendance entre nationalité et survie
contingency_table = pd.crosstab(titanic_data['country'], titanic_data['survived'])
chi2, p, dof, ex = chi2_contingency(contingency_table)
print(f"Chi2 statistic pour la nationalité: {chi2}, p-value: {p}")

# Test de Chi-carré pour l'indépendance entre classe et survie
contingency_table_class = pd.crosstab(titanic_data['class'], titanic_data['survived'])
chi2_class, p_class, dof_class, ex_class = chi2_contingency(contingency_table_class)
print(f"Chi2 statistic pour la classe: {chi2_class}, p-value: {p_class}")



