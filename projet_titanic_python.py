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

# 10. Analyse croisée entre nationalité et classe de billet avec heatmap
pivot_country_class = pd.pivot_table(titanic_data, values='survived', index=['country'], columns=['class'], aggfunc='mean')

plt.figure(figsize=(10, 8))
sns.heatmap(pivot_country_class, annot=True, cmap="Reds", linewidths=0.5)
plt.title('Taux de survie par nationalité et classe')
plt.xticks(rotation=45)
plt.show()

# 11. Diagramme en barres pour examiner la répartition des nationalités
plt.figure(figsize=(32, 10))
sns.countplot(data=titanic_data, x='country', hue='country', palette='Set2')
plt.title('Répartition des passagers par nationalité')
plt.xlabel('Nationalité')
plt.ylabel('Nombre de passagers')
plt.xticks(rotation=45)
plt.show()

# 12. Heatmap pour visualiser le taux de survie par nationalité et sexe
pivot_table_country_gender = pd.pivot_table(titanic_data, values='survived', index='country', columns='gender', aggfunc='mean')

plt.figure(figsize=(10, 8))
sns.heatmap(pivot_table_country_gender, annot=True, cmap='Blues', linewidths=0.5)
plt.title('Taux de survie par nationalité et sexe')
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

#Deux modèles (régression logistique et forêt aléatoire) 

# 1 Modèle Optimisé : Régression Logistique

# Meilleurs paramètres :
#C = 1 : Ce paramètre de régularisation indique qu'une pénalité modérée est appliquée. 
#Une valeur plus grande de `C` signifie moins de régularisation, 
#et une valeur plus petite signifie plus de régularisation. 
#Ici, la régularisation est modérée, ce qui est cohérent avec le compromis entre biais et variance.

#Pénalité = l2 : L'utilisation de la régularisation `L2` (ridge) permet de réduire la magnitude des 
#coefficients pour éviter l'overfitting, mais sans les mettre exactement à zéro (ce que ferait `L1`).

# Performance du modèle :
#Accuracy = 0.7579 (75.79%) : Le modèle a correctement prédit environ 76 % des cas, 
#ce qui est relativement bon mais montre qu'il y a encore une marge d'amélioration.
  
#Matrice de confusion :
#Classe 0 (Non-survivants) : 
#252 passagers ont été correctement classés comme non-survivants (vrais négatifs).
#52 passagers non-survivants ont été mal classés comme survivants (faux positifs).
#Classe 1 (Survivants) : 
#83 passagers survivants ont été correctement identifiés (vrais positifs).
#55 passagers survivants ont été mal classés comme non-survivants (faux négatifs).
  
#Rapport de classification :
#Précision (classe 1) = 0.61 : La précision signifie que 61 % des prédictions positives (survivants) 
#sont effectivement correctes.
#Rappel (classe 1) = 0.60 : Le rappel indique que 60 % des passagers réellement survivants ont été correctement 
#identifiés.
#f1-score (classe 1) = 0.61 : Le F1-score combine la précision et le rappel, et ici, 
#il montre un équilibre modéré pour la classe des survivants.

#Interprétation générale :
#La régression logistique fonctionne relativement bien pour détecter les non-survivants, mais elle a des 
#difficultés à correctement prédire les survivants (classe 1). 
#Cela est visible avec une précision et un rappel modérés pour la classe des survivants.
#Ce modèle a tendance à sous-prévoir les survivants, ce qui est un problème commun lorsqu'on a un 
#déséquilibre de classe, même si des techniques comme `SMOTE` ont été appliquées pour y remédier.



#2 Modèle Optimisé : Forêt Aléatoire (Random Forest)

# Meilleurs paramètres :
#max_depth = None : Cela signifie que les arbres peuvent croître jusqu'à ce que toutes les feuilles soient pures, 
#ou qu'ils contiennent moins d'échantillons que le paramètre `min_samples_split`. 
#Cela permet à chaque arbre de capter un maximum d'informations.
#min_samples_split = 10 : Un nœud doit contenir au moins 10 échantillons avant de se diviser, 
#ce qui limite la croissance excessive des arbres et aide à éviter l'overfitting.
#min_samples_leaf = 1 : Chaque feuille (noeud terminal) doit contenir au moins 1 échantillon, 
#ce qui permet aux arbres de se diviser plus en profondeur.
#n_estimators = 200 : Le modèle utilise 200 arbres pour la prédiction, ce qui est une valeur assez standard, 
#mais plus faible que dans certains modèles de Random Forest plus robustes.

#Performance du modèle :
#Accuracy = 0.7466 (74.66%) : La forêt aléatoire atteint environ 75 % d'accuracy, 
#ce qui est légèrement inférieur à celui de la régression logistique.
  
#Matrice de confusion :
#Classe 0 (Non-survivants) : 
#245 passagers ont été correctement classés comme non-survivants (vrais négatifs).
#59 passagers non-survivants ont été mal classés comme survivants (faux positifs).
#Classe 1 (Survivants): 
#85 passagers survivants ont été correctement identifiés (vrais positifs).
#53 passagers survivants ont été mal classés comme non-survivants (faux négatifs).
  
#Rapport de classification :
#Précision (classe 1) = 0.59 : La précision pour les survivants est légèrement plus faible que dans le modèle 
#de régression logistique, ce qui signifie que 59 % des prédictions positives sont correctes.
#Rappel (classe 1) = 0.62 : Le rappel est légèrement meilleur que dans la régression logistique, 
#avec 62 % des survivants correctement identifiés.
#f1-score (classe 1) = 0.60: Le F1-score est similaire à celui de la régression logistique, 
#montrant un compromis modéré entre précision et rappel.

# Interprétation générale :
#Random Forest montre des performances similaires à celles de la régression logistique en termes d'accuracy 
#globale, mais il gère légèrement mieux le rappel pour les survivants (classe 1). 
#Cela signifie que la forêt aléatoire est légèrement meilleure pour identifier les survivants, 
#bien que la différence soit minime.
#Problème récurrent: Comme pour la régression logistique, le modèle a du mal à correctement prédire 
#la classe des survivants, avec une précision et un rappel encore faibles pour la classe 1.



#Comparaison et Recommandations :

#Précision générale : Les deux modèles présentent des performances globales similaires en termes de précision, 
#avec une légère avance pour la régression logistique.
#Détection des survivants : Les deux modèles ont des difficultés à détecter correctement les survivants, 
#bien que la forêt aléatoire ait un rappel légèrement meilleur pour cette classe.

#Prochaines étapes :
#Rééchantillonnage supplémentaire : Bien que `SMOTE` ait déjà été utilisé pour rééquilibrer les classes, 
#1 il pourrait être utile de tester d'autres techniques, comme l'ajustement des poids de 
#classe (`class_weight='balanced'`).
#2 Ensembles plus robustes avec des modèles plus complexes comme XGBoost ou Gradient Boosting qui pourraient 
#mieux capter les relations complexes entre les variables et améliorer les performances sur la classe 
#des survivants.
#3 Tuning d'hyperparamètres supplémentaire pour améliorer les performances globales, en particulier pour 
#la détection des survivants.


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
#Accuracy : 0.7896
#Précision pour les survivants (classe 1) : 0.70
#Rappel pour les survivants : 0.57
#AUC : 0.81
#Le Gradient Boosting est bien équilibré, et avec son AUC de 0.81, 
#il surpasse légèrement les autres modèles. 
#Cependant, il est toujours un peu en difficulté pour bien prédire les survivants (classe 1), 
#comme en témoigne le rappel à 0.57.

#2. Régression Logistique
#Accuracy : 0.7715
#AUC : 0.78
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

# Importation des bibliothèques à nouveau
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.feature_selection import RFE
from imblearn.over_sampling import SMOTE
from sklearn.metrics import RocCurveDisplay

# Préparation des données pour la modélisation
X = titanic_data.drop(columns=['Num', 'survived'])
y = titanic_data['survived']

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Application de SMOTE pour rééquilibrer les classes
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Standardisation des données
scaler = StandardScaler()
X_resampled_scaled = scaler.fit_transform(X_resampled)
X_test_scaled = scaler.transform(X_test)

# 1. Régression Logistique avec ajustement des poids de classe
logreg = LogisticRegression(solver='lbfgs', max_iter=1000, class_weight='balanced', random_state=42)
param_grid_logreg = {'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l2']}
random_search_logreg = RandomizedSearchCV(logreg, param_distributions=param_grid_logreg, cv=5, scoring='accuracy', n_iter=5, random_state=42)
random_search_logreg.fit(X_resampled_scaled, y_resampled)
best_logreg = random_search_logreg.best_estimator_

# 2. Random Forest avec RandomizedSearchCV et ajustement des poids de classe
rf = RandomForestClassifier(class_weight='balanced', random_state=42)
param_grid_rf = {'n_estimators': [100, 200, 500], 'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
random_search_rf = RandomizedSearchCV(rf, param_distributions=param_grid_rf, cv=5, scoring='accuracy', n_iter=10, random_state=42)
random_search_rf.fit(X_resampled_scaled, y_resampled)
best_rf = random_search_rf.best_estimator_

# 3. XGBoost
xgb = XGBClassifier(random_state=42)
param_grid_xgb = {'n_estimators': [100, 200, 500], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7]}
random_search_xgb = RandomizedSearchCV(xgb, param_distributions=param_grid_xgb, cv=5, scoring='accuracy', n_iter=10, random_state=42)
random_search_xgb.fit(X_resampled_scaled, y_resampled)
best_xgb = random_search_xgb.best_estimator_

# 4. KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_resampled_scaled, y_resampled)

# 5. Ensemble learning - Voting Classifier
ensemble_model = VotingClassifier(estimators=[
    ('logreg', best_logreg), 
    ('rf', best_rf), 
    ('xgb', best_xgb), 
    ('knn', knn)
], voting='soft')  # 'soft' pour utiliser les probabilités
ensemble_model.fit(X_resampled_scaled, y_resampled)

# 6. Validation croisée avec StratifiedKFold pour une meilleure évaluation
skf = StratifiedKFold(n_splits=5)

# Cross-validation pour le modèle de régression logistique
cross_val_scores_logreg = cross_val_score(best_logreg, X_resampled_scaled, y_resampled, cv=skf)
print(f'Logistic Regression CV accuracy: {np.mean(cross_val_scores_logreg):.4f}')

# Cross-validation pour le modèle Random Forest
cross_val_scores_rf = cross_val_score(best_rf, X_resampled_scaled, y_resampled, cv=skf)
print(f'Random Forest CV accuracy: {np.mean(cross_val_scores_rf):.4f}')

# Cross-validation pour le modèle XGBoost
cross_val_scores_xgb = cross_val_score(best_xgb, X_resampled_scaled, y_resampled, cv=skf)
print(f'XGBoost CV accuracy: {np.mean(cross_val_scores_xgb):.4f}')

# Cross-validation pour le modèle de Voting Classifier
cross_val_scores_ensemble = cross_val_score(ensemble_model, X_resampled_scaled, y_resampled, cv=skf)
print(f'Voting Classifier CV accuracy: {np.mean(cross_val_scores_ensemble):.4f}')

# 7. Evaluation sur le jeu de test pour tous les modèles
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


#Interprétations

#1. Régression Logistique (Logistic Regression)
#Accuracy: 0.7557
#AUC: 0.78
#Interprétation:
#La régression logistique a montré une précision de 75.57% avec un AUC de 0.78, indiquant que 
#ce modèle est plutôt efficace pour séparer les survivants et les non-survivants.
#Ce modèle a une précision de 81% pour prédire les non-survivants, mais une précision plus faible de 62% 
#pour les survivants, indiquant des difficultés à identifier correctement les survivants.

#2. Random Forest
#Accuracy: 0.7760
#AUC: 0.79
#Interprétation:
#La Random Forest a légèrement mieux performé que la régression logistique avec une précision de 77.60% 
#et un AUC de 0.79.
#Le modèle Random Forest est plus équilibré avec une meilleure précision (66%) et un rappel (61%) 
#pour les survivants. Cela montre une meilleure capacité à identifier les survivants, 
#bien que ce ne soit pas parfait.

#3. XGBoost
#Accuracy: 0.7624
#AUC: 0.80
#Interprétation:
#XGBoost affiche une précision de 76.24% et un AUC de 0.80, surpassant les modèles précédents 
#en termes de capacité à différencier les classes.
#Ce modèle offre une bonne balance entre précision et rappel, avec une performance légèrement 
#supérieure à la Random Forest dans l'ensemble. Il semble bien adapté pour ce type de problème, 
#bien que le rappel pour les survivants (59%) reste légèrement inférieur à celui de la Random Forest.

#4. Voting Classifier (Ensemble Model)

#Accuracy: 0.7851
#AUC: 0.80
#Interprétation:
#Le classificateur par vote a obtenu la meilleure précision (78.51%) avec un AUC de 0.80, ce qui en fait 
#le meilleur modèle parmi ceux testés.
#Il combine les avantages des différents modèles en votant de manière pondérée sur les probabilités des prédictions. Cela améliore les performances globales, en particulier pour prédire les survivants (précision de 68%).

#Conclusion :
#Le Voting Classifier a montré les meilleures performances globales avec un AUC de 0.80 et une précision de 78.51%. Il combine les forces des autres modèles tout en maintenant un bon compromis entre précision et rappel.
#XGBoost est une excellente option individuelle avec un AUC de 0.80, tout juste derrière le classificateur 
#par vote en termes de précision.
#Random Forest a également montré des performances robustes et stables, avec une bonne capacité à prédire 
#correctement les non-survivants et une bonne précision globale.

