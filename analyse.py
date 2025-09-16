from estimator import define_model
from preprocessing import preprocess
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, classification_report

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Création DataFrame d'analyse
def DataFrame(results: list):
    flat_results = []

    for d in results:
        base = {k: v for k, v in d.items() if k != 'params'}
        params = d['params']
        estimator_params = params.get('estimator_params', {})
    
        # Aplatir tous les niveaux
        flat = {
            **base,
            **params,
            **estimator_params
        }
    
        flat.pop('estimator_params', None)  # Supprime l'entrée nested
        flat_results.append(flat)

    # Créer le DataFrame
    df = pd.DataFrame(flat_results)

    # Colonnes cibles dans l’ordre demandé
    target_columns = [
        'index_set', 'mean_acc', 'std_acc', 'test_accuracy', 'estimator_name',
        'num_layers', 'hidden_size', 'd_model', 'nhead', 'out_channels',
        'dropout', 'weight_decay', 'lr', 'scheduler_factor', 'batch_size'
    ]

    # Ajoute les colonnes manquantes avec NaN
    for col in target_columns:
        if col not in df.columns:
            df[col] = np.nan

    # Réorganise et garde uniquement les colonnes cibles
    df = df[target_columns]

    # Score personnalisé
    df['score'] = df['mean_acc'] - df['std_acc']

    # Trie par score décroissant
    df = df.sort_values(by='score', ascending=False).reset_index(drop=True)    

    return df

# Formattage à l'affichage
def format_value(val, col):
        if pd.isnull(val):
            return '-'
        if col in ['mean_acc', 'std_acc', 'test_accuracy', 'score']:
            return f"{val:.1f}"
        elif col in ['lr', 'weight_decay']:
            return f"{val:.1e}"
        else:
            return val

# Sauvegarde DataFrame d'analyse
def SauvDataFrame(df: pd.DataFrame, save_path: str = None):

    # Création d'une copie formatée pour affichage et export
    df_formatted = df.copy()
    for col in df_formatted.columns:
        df_formatted[col] = df_formatted[col].apply(lambda v: format_value(v, col))

    # Affichage console lisible
    print(df_formatted.to_string(index=False))

    # Export en CSV
    if save_path is not None:
        df_formatted.to_csv(os.path.join(save_path, "resultats_modeles.csv"), index=False)

    return df_formatted

# Plot
def PlotBestResults(df: pd.DataFrame, save_path: str = None):

    #  Afficher seulement les 10 meilleurs jeux d'hyperparamètres
    df = df.sort_values(by='score', ascending=False).head(10)

    indices = df['index_set']
    mean_acc = df['mean_acc']
    std_acc = df['std_acc']
    test_acc = df['test_accuracy']
    
    x = np.arange(len(indices))

    fig, ax = plt.subplots(figsize=(12, 6))

    # Barres verticales centrées sur mean_acc
    bar_width = 0.4
    ax.bar(x, mean_acc, width=bar_width, color='skyblue', label='Mean Accuracy')

    # Traits pour mean ± std (en vertical centré sur x)
    for i in range(len(x)):
        ax.hlines(mean_acc[i] + std_acc[i], x[i] - bar_width/2, x[i] + bar_width/2, color='orange', linewidth=2, label='Mean + Std' if i==0 else "")
        ax.hlines(mean_acc[i] - std_acc[i], x[i] - bar_width/2, x[i] + bar_width/2, color='orange', linewidth=2, label='Mean - Std' if i==0 else "")

    # Trait horizontal pour test_accuracy (un petit trait au milieu de la barre)
    test_line_width = bar_width * 0.6
    for i in range(len(x)):
        ax.hlines(test_acc[i], x[i] - test_line_width/2, x[i] + test_line_width/2, color='green', linewidth=3, label='Test Accuracy' if i==0 else "")

    # Labels et titres
    ax.set_xticks(x)
    ax.set_xticklabels(indices)
    ax.set_xlabel('Index des jeux d\'hyperparamètres')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Top 10 Configurations : mean acc, std, et test accuracy')

    ax.set_ylim(0, 100)
    ax.legend()

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(os.path.join(save_path, "resultats_modeles.png"), dpi=300)
    
    plt.show()

# Confusion Matrix & Classification Report
def ConfusionMatrix(df: pd.DataFrame, save_path: str, index: str | int = "best"):

    if index == "best":
        index = df.iloc[0]['index_set']

    result = results[int(index) - 1] # liste indexée à partir de 0
    model_state = result['final_model']

    features = result['variables']
    input_size = len(features)
    print(f"Nombre de features sélectionnées : {input_size}")
    print(f"Features utilisées : {features}")

    params = result['params']

    model = define_model(params['estimator_name'], params['estimator_params'], input_size).to(device)
    model.load_state_dict(model_state)
    model.eval()

    # # Chargement des données de test
    X_test, y_test = preprocess('../../test.csv', features)

    # # Création du dataloader de test
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=len(X_test), shuffle=False)

    y_true = []
    y_pred = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            outputs = model(X_batch)
            _, predicted = torch.max(outputs, dim=1)

            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(y_batch.cpu().numpy())
    
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred)

    # print(cm)
    # print(report)

    class_names = ['Allongé', 'Assis', 'Debout', 'Marche', 'Descend\nEscalier', 'Monte\nEscalier']

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)

    plt.xlabel('Prédit')
    plt.ylabel('Réel')
    plt.title('Matrice de confusion')

    if save_path is not None:
        plt.savefig(os.path.join(save_path, f"confusion matrix {index}.png"), dpi=300)  
      
    plt.show()


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    save_path = os.getcwd()

    with open(os.path.join(save_path, "results.pkl"), "rb") as f:
        results = pickle.load(f)

    df = DataFrame(results)
    SauvDataFrame(df, save_path)
    PlotBestResults(df, save_path)

    ConfusionMatrix(df, save_path, index="best")
    # ConfusionMatrix(df, save_path, index=244)
