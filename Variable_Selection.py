import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest
from scipy.stats import kruskal
import numpy as np
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

def selectionVariables(csv_file_name: str):
    # Charger les données
    df = pd.read_csv(csv_file_name)
    
    # Obtenir le nom des colonnes (features)
    columns = df.columns.tolist()
    features = df.drop(columns=['Activity']).columns.tolist()
    features_ = df.drop(columns=['Activity', 'subject']).columns.tolist()

    logger.info(f"Nombre initial de variables : {len(features_)}")
    logger.info(f"Variables initiales : {features_}\n")

    # Identifier les groupes de séquences d'activités identiques
    df['Group'] = (df['Activity'] != df['Activity'].shift()).cumsum()

    # Grouper par séquences
    grouped = df.groupby('Group')

    # Extraire X : toutes les colonnes sauf 'Activity' et 'Group'
    X = [group.drop(columns=['Activity', 'Group', 'subject']).values for _, group in grouped]

    # Extraire y : l'activité associée à chaque groupe (la même pour toute la séquence)
    y = [group['Activity'].iloc[0] for _, group in grouped]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    logger.info(f"Sélection initiale des variables:")
    logger.info(f"Méthode de filtrage sur les scores issus de tests de Kruskal-Wallis pour les statistiques mean et std.")

    select_features = []

    X_mean = np.array([np.mean(seq, axis=0) for seq in X])
    X_std = np.array([np.std(seq, axis=0) for seq in X])
    # X_median = np.array([np.median(seq, axis=0) for seq in X])
    # X_min = np.array([np.min(seq, axis=0) for seq in X])
    # X_max = np.array([np.max(seq, axis=0) for seq in X])

    X_list = [X_mean, X_std] #, X_median, X_min, X_max

    for X in X_list:
        group0 = X[y==0]
        group1 = X[y==1]
        group2 = X[y==2]
        group3 = X[y==3]
        group4 = X[y==4]
        group5 = X[y==5]
        selector = SelectKBest(score_func=lambda X, y: np.array([kruskal(group0[:, i], group1[:, i], group2[:, i], group3[:, i], group4[:, i], group5[:, i]).statistic for i in range(X.shape[1])]), k=15)
        X_new = selector.fit_transform(X, y)
        support = selector.get_support()
        kruskal_scores = selector.scores_
        selected_scores = kruskal_scores[support]
        selected_features = np.array(features_)[support]
        sorted_indices = np.argsort(selected_scores)[::-1]

        # plt.figure(figsize=(15,5))
        # plt.bar(x=selected_features[sorted_indices], height=selected_scores[sorted_indices])
        # plt.xticks(rotation=90)
        # plt.xlabel('Caractéristiques')
        # plt.ylabel("H-score")
        # plt.title('Sélection des caractéristiques avec le test de Kruskal-Wallis')
        # plt.tight_layout()
        # plt.show()

        select_features.extend(selected_features.tolist())
        select_features = list(set(select_features))

    return select_features

# select_features = selectionVariables('train.csv')
# print(select_features)
# print(len(select_features))
