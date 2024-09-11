#projet_titanic_python


# Importation des bibliothèques nécessaires
import pandas as pd
import numpy as np

# Charger le fichier Titanic.csv
file_path = 'data/titanic.csv'  # Remplace avec le chemin correct vers ton fichier
titanic_data = pd.read_csv(file_path, sep=';')

# Afficher les informations du dataset pour détecter les valeurs manquantes
missing_values = titanic_data.isnull().sum()

# Afficher les types de données pour vérifier si certaines colonnes nécessitent un formatage
data_types = titanic_data.dtypes

missing_values, data_types

# Suppression de la colonne "ticketno" (car elle est entièrement manquante)
titanic_data.drop(columns=['ticketno'], inplace=True)

# Imputation des valeurs manquantes

# 1. Imputer l'âge avec la médiane
titanic_data['age'].fillna(titanic_data['age'].median(), inplace=True)

# 2. Imputer le prix du billet (fare) avec la moyenne
titanic_data['fare'].fillna(titanic_data['fare'].mean(), inplace=True)

# 3. Imputer la nationalité (country) avec "Unknown"
titanic_data['country'].fillna('Unknown', inplace=True)

# Importer les bibliothèques de visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# Configurer le style de visualisation
sns.set(style="whitegrid")

# 1. Histogramme de la distribution des âges
plt.figure(figsize=(10, 6))
sns.histplot(titanic_data['age'], bins=30, kde=False, color='blue')
plt.title('Distribution des âges des passagers')
plt.xlabel('Âge')
plt.ylabel('Nombre de passagers')
plt.show()

# 2. Diagramme en barres des sexes
plt.figure(figsize=(6, 4))
sns.countplot(data=titanic_data, x='gender', palette='Set2')
plt.title('Répartition des passagers par sexe')
plt.xlabel('Sexe')
plt.ylabel('Nombre de passagers')
plt.show()

# 3. Diagramme en barres de la répartition par classe
plt.figure(figsize=(6, 4))
sns.countplot(data=titanic_data, x='class', palette='Set3')
plt.title('Répartition des passagers par classe')
plt.xlabel('Classe')
plt.ylabel('Nombre de passagers')
plt.show()

# 4. Comparaison des taux de survie par sexe
plt.figure(figsize=(6, 4))
sns.barplot(data=titanic_data, x='gender', y='survived', palette='Set1')
plt.title('Taux de survie par sexe')
plt.xlabel('Sexe')
plt.ylabel('Taux de survie')
plt.show()

# 5. Comparaison des taux de survie par classe
plt.figure(figsize=(6, 4))
sns.barplot(data=titanic_data, x='class', y='survived', palette='Set1')
plt.title('Taux de survie par classe')
plt.xlabel('Classe')
plt.ylabel('Taux de survie')
plt.show()

# 6. Boxplot de l'âge par survie
plt.figure(figsize=(8, 6))
sns.boxplot(data=titanic_data, x='survived', y='age', palette='coolwarm')
plt.title('Âge des passagers par survie')
plt.xlabel('Survécu (0 = Non, 1 = Oui)')
plt.ylabel('Âge')
plt.show()


# Vérification des valeurs manquantes après imputation
print(titanic_data.isnull().sum())

# Affichage des premières lignes pour vérifier les changements
print(titanic_data.head())
