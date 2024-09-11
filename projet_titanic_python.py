#projet_titanic_python


# Importation des bibliothèques nécessaires
import pandas as pd
import numpy as np

# Charger le fichier Titanic.csv
file_path = 'data/titanic.csv'  # Remplace avec le chemin correct vers ton fichier
titanic_data = pd.read_csv(file_path, sep=';')

# Suppression de la colonne "ticketno" (car elle est entièrement manquante)
titanic_data.drop(columns=['ticketno'], inplace=True)

# Imputation des valeurs manquantes

# 1. Imputer l'âge avec la médiane
titanic_data['age'].fillna(titanic_data['age'].median(), inplace=True)

# 2. Imputer le prix du billet (fare) avec la moyenne
titanic_data['fare'].fillna(titanic_data['fare'].mean(), inplace=True)

# 3. Imputer la nationalité (country) avec "Unknown"
titanic_data['country'].fillna('Unknown', inplace=True)

# Vérification des valeurs manquantes après imputation
print(titanic_data.isnull().sum())

# Affichage des premières lignes pour vérifier les changements
print(titanic_data.head())
