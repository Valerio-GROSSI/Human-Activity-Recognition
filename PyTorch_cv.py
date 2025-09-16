from asyncio.log import logger
from preprocessing import preprocess
from estimator import define_model
from Variable_Selection import selectionVariables

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import StratifiedKFold

import random
import numpy as np

import logging
import pickle
import os
import sys
from datetime import datetime

def config_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.hasHandlers():
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = os.path.join("logs", timestamp)
        os.makedirs(log_dir, exist_ok=True)

        log_file_path = os.path.join(log_dir, 'output.log')

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        file_handler = logging.FileHandler(log_file_path, mode='w')
        file_handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return log_dir

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0
    correct =  0
    total = 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * y_batch.size(0)
        _, predicted = torch.max(outputs, dim=1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()
    
    avg_loss = running_loss / total
    accuracy = correct / total * 100
    return avg_loss, accuracy

def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            running_loss += loss.item() * y_batch.size(0)
            _, predicted = torch.max(outputs, dim=1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    
    avg_loss = running_loss / total
    accuracy = correct / total * 100
    return avg_loss, accuracy

def cross_validate_model(X, y, k_folds, num_epochs, patience, batch_size):

    # Configuration de la validation croisée
    kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_state)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
        logger.info(f"=== Fold {fold + 1}/{k_folds} ===")

        # Création des datasets pour ce fold
        train_dataset = TensorDataset(X[train_idx], y[train_idx])
        val_dataset = TensorDataset(X[val_idx], y[val_idx])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

        # Initialisation du modèle pour ce fold
        input_size = X.shape[2]
        model = define_model(estimator_name, estimator_params, input_size).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=scheduler_factor, patience=30)
        
        # Variables pour l'early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        train_losses, val_losses = [], []
        train_accuracies, val_accuracies = [], []
        
        #Entrainement pour ce fold
        for epoch in range(num_epochs):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            if epoch % 200 == 0 or epoch == num_epochs - 1:
                logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - Val loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Early stopping
            if patience_counter >= patience:
                epochs_used = epoch + 1
                logger.info(f"Early stopping à l'epoch {epochs_used}")
                break
            else:
                epochs_used = epoch + 1
        
        # Charger le meilleur modèle
        model.load_state_dict(best_model_state)

        # Évaluation finale sur la validation
        final_val_loss, final_val_acc = validate_epoch(model, val_loader, criterion, device)

        fold_results.append({
            'fold': fold + 1,
            'val_accuracy': final_val_acc,
            'val_loss': final_val_loss,
            'epochs_used': epochs_used,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies,
            'best_model_state': best_model_state
        })

        logger.info(f"Fold {fold + 1} - Accuracy finale: {final_val_acc:.2f}%")

    return fold_results
        
def evaluate_cross_validation_results(fold_results, set_idx):
    val_accuracies = [result['val_accuracy'] for result in fold_results]
    val_losses = [result['val_loss'] for result in fold_results]
    epoch_counts = [result['epochs_used'] for result in fold_results]
    
    mean_accuracy = np.mean(val_accuracies)
    std_accuracy = np.std(val_accuracies)
    mean_loss = np.mean(val_losses)
    std_loss = np.std(val_losses)
    avg_epochs_used = int(np.mean(epoch_counts))

    logger.info(f"Accuracy moyenne: {mean_accuracy:.2f}% ± {std_accuracy:.2f}%")
    logger.info(f"Loss moyenne: {mean_loss:.4f} ± {std_loss:.4f}")
    logger.info(f"Nombre d'epochs moyen: {avg_epochs_used}")
    logger.info(f"Accuracy par fold: {[f'{acc:.2f}%' for acc in val_accuracies]}\n")
    
    return mean_accuracy, std_accuracy, avg_epochs_used

def hyperparameter_set_results(X_train, y_train, test_loader, set_idx, params, features, use_forward_selection):

    if not use_forward_selection:
        logger.info(f"Début de la validation croisée pour le jeu d'hyperparamètres {set_idx + 1}\n")
        logger.info(f"Variables sélectionnées : mêmes variables que celles retenues au départ (pas de sélection en avant)\n")
        fold_results = cross_validate_model(X_train, y_train, n_splits, num_epochs, patience, batch_size)
        
        # Évaluation des résultats de la validation croisée
        logger.info("")
        logger.info(f"Résultats de la validation croisée")
        logger.info("")
        mean_acc, std_acc, avg_epochs_used = evaluate_cross_validation_results(fold_results, set_idx)

        X_train_final = X_train
        best_fold_results = fold_results
        best_avg_epochs_used = avg_epochs_used
        best_mean_acc = mean_acc
        best_std_acc = std_acc
        input_size_final = X_train.shape[2]
        selected_features = list(range(input_size_final))
        test_loader_final = test_loader
    else:
        logger.info(f"Début des validations croisées pour le jeu d'hyperparamètres {set_idx + 1}\n")
        logger.info(f"Sélection de variables en avant")
        logger.info(f"Score considéré : mean accuracy - std accuracy\n")
        selected_features = []
        remaining_features = list(range(X_train.shape[2]))
        best_score = -float('inf')
        max_features = len(features) # len(features) <=> None pour ne pas limiter

        while len(remaining_features) > 0 and (max_features is None or len(selected_features) < max_features):
            scores = []

            logger.info(f"Variables actuellement selectionnées : {[features[i] for i in selected_features]}\n")
            for f in remaining_features:
                logger.info(f"Variable candidate {features[f]} testée")
                logger.info("")
                current_features = selected_features + [f]

                X_train_sub = torch.tensor(X_train[:, :, current_features], dtype=torch.float32)

                # Validation croisée
                fold_results = cross_validate_model(X_train_sub, y_train, n_splits, num_epochs, patience, batch_size)
    
                # Évaluation des résultats de la validation croisée
                logger.info("")
                logger.info(f"Résultats de la validation croisée")
                logger.info("")
                mean_acc, std_acc, avg_epochs_used = evaluate_cross_validation_results(fold_results, set_idx)
                score = mean_acc - std_acc
                scores.append((f, score, mean_acc, std_acc, fold_results, avg_epochs_used))

            # Meilleure variable à ajouter
            f_best, score_best, mean_acc_best, std_acc_best, fold_results_best, avg_epochs_used_best = max(scores, key=lambda x: x[1])

            if score_best > best_score:
                selected_features.append(f_best)
                remaining_features.remove(f_best)
                best_score = score_best
                best_fold_results = fold_results_best
                best_mean_acc = mean_acc_best
                best_std_acc = std_acc_best
                best_avg_epochs_used = avg_epochs_used_best
                logger.info(f"Ajout de la variable {features[f_best]}, mean accuracy - std accuracy: {score_best:.4f}\n")
            else:
                logger.info("Plus d'amélioration -> arret\n")
                break

        logger.info(f"Variables sélectionnées : {[features[i] for i in selected_features]}")
        logger.info("")
        logger.info(f"Résultats de la validation croisée pour le jeu d'hyperparamètres {set_idx + 1}")
        logger.info("")
        evaluate_cross_validation_results(best_fold_results, set_idx)

        X_train_final = X_train[:, :, selected_features]
        input_size_final = X_train_final.shape[2]

        X_test, y_test = next(iter(test_loader))
        X_test_final = X_test[:, :, selected_features]
        test_dataset = TensorDataset(X_test_final, y_test)
        test_loader_final = DataLoader(test_dataset, batch_size=len(X_test_final), shuffle=False)

    # Entraînement du modèle final sur toutes les données d'entraînement
    logger.info(f"Entrainement du modèle final pour le jeu d'hyperparamètres {set_idx + 1}\n")
    
    # Création des datasets
    full_train_dataset = TensorDataset(X_train_final, y_train)
    full_train_loader = DataLoader(full_train_dataset, batch_size=batch_size, shuffle=True)

    # Initialisation du modèle
    model = define_model(estimator_name, estimator_params, input_size_final).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(best_avg_epochs_used):
        train_loss, train_acc = train_epoch(model, full_train_loader, criterion, optimizer, device)
        
        if epoch % 200 == 0 or epoch == best_avg_epochs_used - 1:
            logger.info(f"Epoch {epoch+1}/{best_avg_epochs_used} - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
    
    # Inférence du modèle final sur les données de test
    test_loss, test_accuracy = validate_epoch(model, test_loader_final, criterion, device)

    final_model_state = model.state_dict().copy()

    logger.info(f"\n")
    logger.info(f"Résultats finaux pour le jeu d'hyperparamètres {set_idx + 1}\n")
    logger.info(f"Accuracy de validation croisée: {best_mean_acc:.2f}% ± {best_std_acc:.2f}%")
    logger.info(f"Accuracy sur le test: {test_accuracy:.2f}%\n")

    dico = {'index_set': set_idx + 1,
            'params': params,
            'variables': [features[i] for i in selected_features] if use_forward_selection else features,
            'mean_acc': best_mean_acc,
            'std_acc': best_std_acc,
            'test_accuracy': test_accuracy,
            'fold_results': best_fold_results,
            'final_model': final_model_state
            }

    return dico

# Main execution
if __name__ == "__main__":

    log_dir = config_logger()
    logger = logging.getLogger()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Nombre de classes du jeu de données
    logger.info(f"ANALYSE DES CLASSES\n")
    num_classes = 6
    logger.info(f"Les données sont réparties en {num_classes} classes")
    logger.info(f"Classes: LAYING, SITTING, STANDING, WALKING, WALKING_DOWNSTAIRS, WALKING_UPSTAIRS\n")

    # Features du jeu de données à traiter
    logger.info(f"ANALYSE DES VARIABLES\n")
    # features = ['tBodyAcc-mean()-X', 'tBodyAcc-mean()-Y', 'tBodyAcc-mean()-Z',
    # 'tBodyGyro-mean()-X', 'tBodyGyro-mean()-Y', 'tBodyGyro-mean()-Z']
    features = selectionVariables('train.csv')

    logger.info(f"Nombre de variables retenues : {len(features)}")
    logger.info(f"Variables retenues : {features}\n")
    
    # Chargement des données d'entrainement
    logger.info("ANALYSE DES DONNEES D'ENTRAINEMENT\n")
    X_train, y_train = preprocess('train.csv', features)

    # Chargement des données de test
    logger.info("ANALYSE DES DONNEES DE TEST\n")
    X_test, y_test = preprocess('test.csv', features)

    # Création du dataloader de test
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=len(X_test), shuffle=False)

    # Utilisation de la sélection de variables en avant lors de la validation croisée
    USE_FORWARD_SELECTION = True

    ## Hyperparamètres fixés
    num_epochs = 500 # nb cycles d'entrainement maximal
    patience = 80 # arret précoce sur la validation loss

    # Stratified k-fold cross-validation
    n_splits = 10
    random_state = 42 # folders identiques entre chaque jeu d'hyperparamètres testés

    # Métrique: CrossEntropyLoss
    # Optimiseur: optim.Adam comprenant weight_decay
    # Planificateur: optim.lr_scheduler.ReduceLROnPlateau avec mode='min' et patience = 30

    ## Hyperparamètres testés
    # classifieur : 
        # LSTM: LSTM + couche de sortie linéaire comprenant dropout
            # num_layers, hidden_size, dropout
        # CNN + LSTM: CNN + LSTM + couche de sortie linéaire comprenant dropout
            # out_channels, num_layers, hidden_size, dropout
        # Transformer: Transformer comprenant dropout
            # d_model, nhead, num_layers, dropout
    
    # weight_decay
    # lr
    # scheduler_factor
    # batch_size

    param_grid = {
        'estimator': {
            # 'LSTM': {
            #     'num_layers': [2, 3, 4], 
            #     'hidden_size': [64, 80, 96],
            #     'dropout': [0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175]
            # },
            # 'CNN + LSTM': {
            #     'out_channels': [8, 16, 32], 
            #     'num_layers': [2, 3, 4], 
            #     'hidden_size': [64, 80, 96],
            #     'dropout': [0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175]
            # }, 
            'Transformer': {
                # 'd_model': [36, 48, 60, 72],
                'd_model': [36],
                # 'nhead': [2, 3, 4, 12],
                'nhead': [3],
                # 'num_layers': [2, 3, 4],
                'num_layers': [2],
                # 'dropout': [0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175]
                'dropout': [0.125]
            }
        },
        # 'weight_decay': [5e-7, 7.5e-7, 1e-6, 2.5e-6, 5e-6],
        'weight_decay': [1e-6],
        # 'lr':  [3.5e-3, 4e-3, 4.5e-3, 5e-3, 5.5e-3],
        'lr':  [4e-3],
        # 'scheduler_factor': [0.4, 0.5, 0.6],
        'scheduler_factor': [0.6],
        # 'batch_size': [28, 32, 36]
        'batch_size': [36]
    }
    logger.info(f"HYPERPARAMETRES FIXES\n")
    logger.info(f"Nombre de cycles d'entrainement maximal: {num_epochs}")
    logger.info(f"Nombre de cycles d'entrainement maximal sans amélioration avant arret précoce: {patience}")
    logger.info(f"Planificateur du taux d'apprentissage: ReduceLROnPlateau")
    logger.info(f"Nombre d'époques de tolérance du planificateur: {30}")
    logger.info(f"Nombre de folds pour la validation croisée stratifiée: {n_splits}")
    logger.info(f"Optimiseur: Adam avec weight_decay")
    logger.info(f"Métrique de perte: CrossEntropyLoss")
    logger.info(f"Graine aléatoire pour la reproductibilité des découpages de la validation croisée entre chaque jeu d'hyperparamètres: {random_state}\n")

    logger.info(f"HYPERPARAMETRES TESTES\n")
    logger.info(f"{param_grid}\n")

    logger.info(f"DÉBUT DE LA RECHERCHE ALEATOIRE D'HYPERPARAMÈTRES AVEC SAUVEGARDE DES RESULTATS\n")
    n_trials = 1
    logger.info(f"Nombre de jeux d'hyperparamètres à tester: {n_trials}\n")

    results = [] # contiendra tous les résultats

    for set_idx in range(n_trials):
        estimator_name = random.choice(list(param_grid['estimator'].keys()))
        estimator_params = {}

        if estimator_name == 'LSTM':
            estimator_params = {
                'num_layers': random.choice(param_grid['estimator']['LSTM']['num_layers']),
                'hidden_size': random.choice(param_grid['estimator']['LSTM']['hidden_size']),
                'dropout': random.choice(param_grid['estimator']['LSTM']['dropout']),
            }
        elif estimator_name == 'CNN + LSTM':
            estimator_params = {
                'out_channels': random.choice(param_grid['estimator']['CNN + LSTM']['out_channels']),
                'num_layers': random.choice(param_grid['estimator']['CNN + LSTM']['num_layers']),
                'hidden_size': random.choice(param_grid['estimator']['CNN + LSTM']['hidden_size']),
                'dropout': random.choice(param_grid['estimator']['CNN + LSTM']['dropout']),
            }
        elif estimator_name == 'Transformer':
            estimator_params = {
                'd_model': random.choice(param_grid['estimator']['Transformer']['d_model']),
                'nhead': random.choice(param_grid['estimator']['Transformer']['nhead']),
                'num_layers': random.choice(param_grid['estimator']['Transformer']['num_layers']),
                'dropout': random.choice(param_grid['estimator']['Transformer']['dropout']),
            }

        weight_decay = random.choice(param_grid['weight_decay'])
        lr = random.choice(param_grid['lr'])
        scheduler_factor = random.choice(param_grid['scheduler_factor'])
        batch_size = random.choice(param_grid['batch_size'])

        params = {
            'estimator_name': estimator_name,
            'estimator_params': estimator_params,
            'weight_decay': weight_decay,
            'lr': lr,
            'scheduler_factor': scheduler_factor,
            'batch_size': batch_size
        }

        logger.info(f"JEU D'HYPERPARAMÈTRES {set_idx + 1}:")
        logger.info(f"{params}\n")

        dico = hyperparameter_set_results(X_train, y_train, test_loader, set_idx, params, features, USE_FORWARD_SELECTION)
        results.append(dico)

        with open(os.path.join(log_dir, "results.pkl"), "wb") as f:
            pickle.dump(results, f)
