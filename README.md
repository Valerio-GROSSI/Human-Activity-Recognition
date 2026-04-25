# Human Activity Recognition

This project focuses on Human Activity Recognition (HAR) using multivariate time-series data collected from smartphone IMU sensors.

It aims to classify different types of physical activities by training and evaluating supervised machine learning models on labeled datasets.

Please refer to the PDF in the current directory for more detailed information about this project and instructions on how to run it.
For the first run and setup, please refer to the instructions below.

<br>

## First Run

1. To have a python environment containing the expected libraries for running the scripts :
```bash
conda create -n name python=3.10
conda activate name
pip install -r requirements.txt
```

2. Download the CSV files (Kaggle dataset), with this command :
```bash
python Import_Data.py
```

4. Move the downloaded CSV files in the project folder. The path were the CSV files were downloaded is displayed in the terminal.

3. Execute the command :
```bash
python PyTorch_cv.py
```

5. Execute the following command in the results folder that was just created by the previous command :
```bash
cd logs/...
python analyse.py
```

<br>

## Sample Results

Sample outputs on the Kaggle dataset (Confusion matrix on the Test set) :

<p align="center">
<img src="logs/2025-09-01_19-14-30/confusion matrix 27.png" width="80%"><br>
<b>Optimal classifier using the 6 main variables</b><br>
Validation score: 96.8% ± 1.9%. Test score: 86.7%<br>
</p>

<p align="center">
<img src="logs/2025-09-03_18-21-58/confusion matrix 1.png" width="80%"><br>
<b>Optimal classifier using the full feature set (7 input variables)</b><br>
Validation score: 95.4% ± 2.3%. Test score: 95%<br>
</p>

<br>

<br>

<p align="center">
<b>Training log of the optimal classifier using the 6 main variables (see /logs) :</b><br>
</p>

```bash
2025-09-01 19:14:30,573 - ANALYSE DES CLASSES

2025-09-01 19:14:30,573 - Les données sont réparties en 6 classes
2025-09-01 19:14:30,573 - Classes: LAYING, SITTING, STANDING, WALKING, WALKING_DOWNSTAIRS, WALKING_UPSTAIRS

2025-09-01 19:14:30,573 - ANALYSE DES VARIABLES

2025-09-01 19:14:30,573 - Nombre de variables retenues : 6
2025-09-01 19:14:30,573 - Variables retenues : ['tBodyAcc-mean()-X', 'tBodyAcc-mean()-Y', 'tBodyAcc-mean()-Z', 'tBodyGyro-mean()-X', 'tBodyGyro-mean()-Y', 'tBodyGyro-mean()-Z']

2025-09-01 19:14:30,573 - ANALYSE DES DONNEES D'ENTRAINEMENT

2025-09-01 19:14:31,015 - Il y a 280 séquences de mesures
2025-09-01 19:14:31,033 - Répartition des classes:
2025-09-01 19:14:31,035 - LAYING : 43 séquences (15.36%)
2025-09-01 19:14:31,035 - SITTING : 43 séquences (15.36%)
2025-09-01 19:14:31,035 - STANDING : 42 séquences (15.00%)
2025-09-01 19:14:31,035 - WALKING : 42 séquences (15.00%)
2025-09-01 19:14:31,035 - WALKING_DOWNSTAIRS : 56 séquences (20.00%)
2025-09-01 19:14:31,035 - WALKING_UPSTAIRS : 54 séquences (19.29%)
2025-09-01 19:14:31,047 - La plus longue séquence comporte 48 éléments

2025-09-01 19:14:31,054 - Les données sont centrées-réduites et un padding de zeros est appliqué afin d'uniformiser la longueur des séquences à celle maximale

2025-09-01 19:14:31,054 - ANALYSE DES DONNEES DE TEST

2025-09-01 19:14:31,195 - Il y a 120 séquences de mesures
2025-09-01 19:14:31,195 - Répartition des classes:
2025-09-01 19:14:31,195 - LAYING : 18 séquences (15.00%)
2025-09-01 19:14:31,195 - SITTING : 18 séquences (15.00%)
2025-09-01 19:14:31,195 - STANDING : 18 séquences (15.00%)
2025-09-01 19:14:31,195 - WALKING : 18 séquences (15.00%)
2025-09-01 19:14:31,195 - WALKING_DOWNSTAIRS : 25 séquences (20.83%)
2025-09-01 19:14:31,195 - WALKING_UPSTAIRS : 23 séquences (19.17%)
2025-09-01 19:14:31,200 - La plus longue séquence comporte 40 éléments

2025-09-01 19:14:31,201 - Les données sont centrées-réduites et un padding de zeros est appliqué afin d'uniformiser la longueur des séquences à celle maximale

2025-09-01 19:14:31,201 - HYPERPARAMETRES FIXES

2025-09-01 19:14:31,201 - Nombre de cycles d'entrainement maximal: 500
2025-09-01 19:14:31,201 - Nombre de cycles d'entrainement maximal sans amélioration avant arret précoce: 80
2025-09-01 19:14:31,201 - Planificateur du taux d'apprentissage: ReduceLROnPlateau
2025-09-01 19:14:31,201 - Nombre d'époques de tolérance du planificateur: 30
2025-09-01 19:14:31,201 - Nombre de folds pour la validation croisée stratifiée: 10
2025-09-01 19:14:31,201 - Optimiseur: Adam avec weight_decay
2025-09-01 19:14:31,201 - Métrique de perte: CrossEntropyLoss
2025-09-01 19:14:31,201 - Graine aléatoire pour la reproductibilité des découpages de la validation croisée entre chaque jeu d'hyperparamètres: 42

2025-09-01 19:14:31,201 - HYPERPARAMETRES TESTES

2025-09-01 19:14:31,201 - {'estimator': {'Transformer': {'d_model': [36, 48, 60, 72], 'nhead': [2, 3, 4, 12], 'num_layers': [2, 3, 4], 'dropout': [0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175]}}, 'weight_decay': [5e-07, 7.5e-07, 1e-06, 2.5e-06, 5e-06], 'lr': [0.0035, 0.004, 0.0045, 0.005, 0.0055], 'scheduler_factor': [0.4, 0.5, 0.6], 'batch_size': [28, 32, 36]}

2025-09-01 19:14:31,201 - DÉBUT DE LA RECHERCHE ALEATOIRE D'HYPERPARAMÈTRES AVEC SAUVEGARDE DES RESULTATS

2025-09-01 19:14:31,201 - Nombre de jeux d'hyperparamètres à tester: 300

2025-09-01 19:14:31,201 - JEU D'HYPERPARAMÈTRES 1:
2025-09-01 19:14:31,201 - {'estimator_name': 'Transformer', 'estimator_params': {'d_model': 72, 'nhead': 4, 'num_layers': 4, 'dropout': 0.075}, 'weight_decay': 2.5e-06, 'lr': 0.0035, 'scheduler_factor': 0.4, 'batch_size': 32}

2025-09-01 19:14:31,201 - Début de la validation croisée pour le jeu d'hyperparamètres 1

2025-09-01 19:14:31,201 - Variables sélectionnées : mêmes variables que celles retenues au départ (pas de sélection en avant)

2025-09-01 19:14:31,203 - === Fold 1/10 ===
2025-09-01 19:14:32,997 - Epoch 1/500 - Train Loss: 2.2143, Train Acc: 15.08% - Val loss: 1.8508, Val Acc: 17.86%
2025-09-01 19:14:35,744 - Early stopping à l'epoch 86
2025-09-01 19:14:35,745 - Fold 1 - Accuracy finale: 21.43%
2025-09-01 19:14:35,746 - === Fold 2/10 ===
2025-09-01 19:14:35,793 - Epoch 1/500 - Train Loss: 2.2919, Train Acc: 15.48% - Val loss: 1.9059, Val Acc: 17.86%
2025-09-01 19:14:38,610 - Early stopping à l'epoch 89
2025-09-01 19:14:38,612 - Fold 2 - Accuracy finale: 21.43%
2025-09-01 19:14:38,612 - === Fold 3/10 ===
2025-09-01 19:14:38,651 - Epoch 1/500 - Train Loss: 2.3582, Train Acc: 15.87% - Val loss: 1.8260, Val Acc: 21.43%
2025-09-01 19:14:41,378 - Early stopping à l'epoch 87
2025-09-01 19:14:41,380 - Fold 3 - Accuracy finale: 21.43%
2025-09-01 19:14:41,380 - === Fold 4/10 ===
2025-09-01 19:14:41,418 - Epoch 1/500 - Train Loss: 2.2608, Train Acc: 16.27% - Val loss: 1.9493, Val Acc: 17.86%
2025-09-01 19:14:44,251 - Early stopping à l'epoch 88
2025-09-01 19:14:44,253 - Fold 4 - Accuracy finale: 21.43%
2025-09-01 19:14:44,253 - === Fold 5/10 ===
2025-09-01 19:14:44,293 - Epoch 1/500 - Train Loss: 2.3225, Train Acc: 16.67% - Val loss: 1.9863, Val Acc: 14.29%
2025-09-01 19:14:47,013 - Early stopping à l'epoch 85
2025-09-01 19:14:47,015 - Fold 5 - Accuracy finale: 21.43%
2025-09-01 19:14:47,015 - === Fold 6/10 ===
2025-09-01 19:14:47,053 - Epoch 1/500 - Train Loss: 2.1476, Train Acc: 17.06% - Val loss: 1.8863, Val Acc: 17.86%
2025-09-01 19:14:49,646 - Early stopping à l'epoch 83
2025-09-01 19:14:49,648 - Fold 6 - Accuracy finale: 21.43%
2025-09-01 19:14:49,648 - === Fold 7/10 ===
2025-09-01 19:14:49,689 - Epoch 1/500 - Train Loss: 2.2805, Train Acc: 15.48% - Val loss: 1.8451, Val Acc: 14.29%
2025-09-01 19:14:52,590 - Early stopping à l'epoch 87
2025-09-01 19:14:52,592 - Fold 7 - Accuracy finale: 17.86%
2025-09-01 19:14:52,592 - === Fold 8/10 ===
2025-09-01 19:14:52,631 - Epoch 1/500 - Train Loss: 2.1117, Train Acc: 15.48% - Val loss: 1.8606, Val Acc: 14.29%
2025-09-01 19:14:55,323 - Early stopping à l'epoch 85
2025-09-01 19:14:55,325 - Fold 8 - Accuracy finale: 17.86%
2025-09-01 19:14:55,325 - === Fold 9/10 ===
2025-09-01 19:14:55,367 - Epoch 1/500 - Train Loss: 2.2645, Train Acc: 16.27% - Val loss: 1.8233, Val Acc: 21.43%
2025-09-01 19:14:58,657 - Early stopping à l'epoch 99
2025-09-01 19:14:58,659 - Fold 9 - Accuracy finale: 17.86%
2025-09-01 19:14:58,659 - === Fold 10/10 ===
2025-09-01 19:14:58,701 - Epoch 1/500 - Train Loss: 2.2384, Train Acc: 21.43% - Val loss: 1.9112, Val Acc: 17.86%
2025-09-01 19:15:01,577 - Early stopping à l'epoch 86
2025-09-01 19:15:01,579 - Fold 10 - Accuracy finale: 17.86%
2025-09-01 19:15:01,579 - 
2025-09-01 19:15:01,579 - Résultats de la validation croisée
2025-09-01 19:15:01,579 - 
2025-09-01 19:15:01,579 - Accuracy moyenne: 20.00% ± 1.75%
2025-09-01 19:15:01,579 - Loss moyenne: 1.7850 ± 0.0013
2025-09-01 19:15:01,579 - Nombre d'epochs moyen: 87
2025-09-01 19:15:01,579 - Accuracy par fold: ['21.43%', '21.43%', '21.43%', '21.43%', '21.43%', '21.43%', '17.86%', '17.86%', '17.86%', '17.86%']

2025-09-01 19:15:01,579 - Entrainement du modèle final pour le jeu d'hyperparamètres 1

2025-09-01 19:15:01,624 - Epoch 1/87 - Loss: 2.2339, Acc: 17.14%
2025-09-01 19:15:04,646 - Epoch 87/87 - Loss: 1.7896, Acc: 18.57%
2025-09-01 19:15:04,648 - 

2025-09-01 19:15:04,648 - Résultats finaux pour le jeu d'hyperparamètres 1

2025-09-01 19:15:04,648 - Accuracy de validation croisée: 20.00% ± 1.75%
2025-09-01 19:15:04,648 - Accuracy sur le test: 20.83%

...

2025-09-01 23:28:08,619 - JEU D'HYPERPARAMÈTRES 300:
...
```

<br>

<p align="center">
<b>Training log of the optimal classifier using the full feature set (see /logs) :</b><br>
</p>

```bash
2025-09-03 18:21:58,124 - ANALYSE DES CLASSES

2025-09-03 18:21:58,124 - Les données sont réparties en 6 classes
2025-09-03 18:21:58,124 - Classes: LAYING, SITTING, STANDING, WALKING, WALKING_DOWNSTAIRS, WALKING_UPSTAIRS

2025-09-03 18:21:58,124 - ANALYSE DES VARIABLES

2025-09-03 18:21:58,873 - Nombre initial de variables : 561
2025-09-03 18:21:58,873 - Variables initiales : ['tBodyAcc-mean()-X', ...]

2025-09-03 18:21:58,936 - Sélection initiale des variables:
2025-09-03 18:21:58,936 - Méthode de filtrage sur les scores issus de tests de Kruskal-Wallis pour les statistiques mean et std.
2025-09-03 18:21:59,293 - Nombre de variables retenues : 26
2025-09-03 18:21:59,293 - Variables retenues : ['fBodyAcc-bandsEnergy()-1,16', ..., 'tBodyGyro-mean()-X', ..., 'tBodyAcc-std()-X']

2025-09-03 18:21:59,293 - ANALYSE DES DONNEES D'ENTRAINEMENT

2025-09-03 18:21:59,628 - Il y a 280 séquences de mesures
2025-09-03 18:21:59,775 - Répartition des classes:
2025-09-03 18:21:59,790 - LAYING : 43 séquences (15.36%)
2025-09-03 18:21:59,790 - SITTING : 43 séquences (15.36%)
2025-09-03 18:21:59,790 - STANDING : 42 séquences (15.00%)
2025-09-03 18:21:59,790 - WALKING : 42 séquences (15.00%)
2025-09-03 18:21:59,790 - WALKING_DOWNSTAIRS : 56 séquences (20.00%)
2025-09-03 18:21:59,790 - WALKING_UPSTAIRS : 54 séquences (19.29%)
2025-09-03 18:21:59,810 - La plus longue séquence comporte 48 éléments

2025-09-03 18:21:59,856 - Les données sont centrées-réduites et un padding de zeros est appliqué afin d'uniformiser la longueur des séquences à celle maximale

2025-09-03 18:21:59,857 - ANALYSE DES DONNEES DE TEST

2025-09-03 18:22:00,303 - Il y a 120 séquences de mesures
2025-09-03 18:22:00,303 - Répartition des classes:
2025-09-03 18:22:00,303 - LAYING : 18 séquences (15.00%)
2025-09-03 18:22:00,303 - SITTING : 18 séquences (15.00%)
2025-09-03 18:22:00,303 - STANDING : 18 séquences (15.00%)
2025-09-03 18:22:00,303 - WALKING : 18 séquences (15.00%)
2025-09-03 18:22:00,303 - WALKING_DOWNSTAIRS : 25 séquences (20.83%)
2025-09-03 18:22:00,303 - WALKING_UPSTAIRS : 23 séquences (19.17%)
2025-09-03 18:22:00,309 - La plus longue séquence comporte 40 éléments

2025-09-03 18:22:00,309 - Les données sont centrées-réduites et un padding de zeros est appliqué afin d'uniformiser la longueur des séquences à celle maximale

2025-09-03 18:22:00,310 - HYPERPARAMETRES FIXES

2025-09-03 18:22:00,310 - Nombre de cycles d'entrainement maximal: 500
2025-09-03 18:22:00,310 - Nombre de cycles d'entrainement maximal sans amélioration avant arret précoce: 80
2025-09-03 18:22:00,310 - Planificateur du taux d'apprentissage: ReduceLROnPlateau
2025-09-03 18:22:00,310 - Nombre d'époques de tolérance du planificateur: 30
2025-09-03 18:22:00,310 - Nombre de folds pour la validation croisée stratifiée: 10
2025-09-03 18:22:00,310 - Optimiseur: Adam avec weight_decay
2025-09-03 18:22:00,310 - Métrique de perte: CrossEntropyLoss
2025-09-03 18:22:00,310 - Graine aléatoire pour la reproductibilité des découpages de la validation croisée entre chaque jeu d'hyperparamètres: 42

2025-09-03 18:22:00,310 - HYPERPARAMETRES TESTES

2025-09-03 18:22:00,310 - {'estimator': {'Transformer': {'d_model': [36], 'nhead': [3], 'num_layers': [2], 'dropout': [0.125]}}, 'weight_decay': [1e-06], 'lr': [0.004], 'scheduler_factor': [0.6], 'batch_size': [36]}

2025-09-03 18:22:00,310 - DÉBUT DE LA RECHERCHE ALEATOIRE D'HYPERPARAMÈTRES AVEC SAUVEGARDE DES RESULTATS

2025-09-03 18:22:00,310 - Nombre de jeux d'hyperparamètres à tester: 1

2025-09-03 18:22:00,310 - JEU D'HYPERPARAMÈTRES 1:
2025-09-03 18:22:00,310 - {'estimator_name': 'Transformer', 'estimator_params': {'d_model': 36, 'nhead': 3, 'num_layers': 2, 'dropout': 0.125}, 'weight_decay': 1e-06, 'lr': 0.004, 'scheduler_factor': 0.6, 'batch_size': 36}

2025-09-03 18:22:00,310 - Début des validations croisées pour le jeu d'hyperparamètres 1

2025-09-03 18:22:00,310 - Sélection de variables en avant
2025-09-03 18:22:00,310 - Score considéré : mean accuracy - std accuracy

2025-09-03 18:22:00,310 - Variables actuellement selectionnées : []

2025-09-03 18:22:00,310 - Variable candidate fBodyAcc-bandsEnergy()-1,16 testée
2025-09-03 18:22:00,310 - 
2025-09-03 18:22:00,346 - === Fold 1/10 ===
2025-09-03 18:22:07,426 - Epoch 1/500 - Train Loss: 1.3703, Train Acc: 40.48% - Val loss: 1.0500, Val Acc: 50.00%
2025-09-03 18:22:09,349 - Early stopping à l'epoch 125
2025-09-03 18:22:09,351 - Fold 1 - Accuracy finale: 82.14%
2025-09-03 18:22:09,351 - === Fold 2/10 ===
2025-09-03 18:22:09,373 - Epoch 1/500 - Train Loss: 1.3928, Train Acc: 34.52% - Val loss: 1.1644, Val Acc: 35.71%
2025-09-03 18:22:10,833 - Early stopping à l'epoch 97
2025-09-03 18:22:10,834 - Fold 2 - Accuracy finale: 67.86%
2025-09-03 18:22:10,834 - === Fold 3/10 ===
2025-09-03 18:22:10,853 - Epoch 1/500 - Train Loss: 1.3769, Train Acc: 33.33% - Val loss: 1.1189, Val Acc: 46.43%
2025-09-03 18:22:13,147 - Early stopping à l'epoch 152
2025-09-03 18:22:13,148 - Fold 3 - Accuracy finale: 64.29%
2025-09-03 18:22:13,148 - === Fold 4/10 ===
2025-09-03 18:22:13,166 - Epoch 1/500 - Train Loss: 1.4367, Train Acc: 34.92% - Val loss: 1.0135, Val Acc: 46.43%
2025-09-03 18:22:14,571 - Early stopping à l'epoch 93
2025-09-03 18:22:14,572 - Fold 4 - Accuracy finale: 57.14%
2025-09-03 18:22:14,573 - === Fold 5/10 ===
2025-09-03 18:22:14,591 - Epoch 1/500 - Train Loss: 1.3431, Train Acc: 40.87% - Val loss: 1.1252, Val Acc: 50.00%
2025-09-03 18:22:16,157 - Early stopping à l'epoch 104
2025-09-03 18:22:16,158 - Fold 5 - Accuracy finale: 46.43%
2025-09-03 18:22:16,158 - === Fold 6/10 ===
2025-09-03 18:22:16,177 - Epoch 1/500 - Train Loss: 1.4905, Train Acc: 35.32% - Val loss: 1.1366, Val Acc: 46.43%
2025-09-03 18:22:18,146 - Early stopping à l'epoch 130
2025-09-03 18:22:18,147 - Fold 6 - Accuracy finale: 57.14%
2025-09-03 18:22:18,147 - === Fold 7/10 ===
2025-09-03 18:22:18,167 - Epoch 1/500 - Train Loss: 1.3779, Train Acc: 39.29% - Val loss: 1.1661, Val Acc: 39.29%
2025-09-03 18:22:20,986 - Early stopping à l'epoch 187
2025-09-03 18:22:20,988 - Fold 7 - Accuracy finale: 64.29%
2025-09-03 18:22:20,988 - === Fold 8/10 ===
2025-09-03 18:22:21,006 - Epoch 1/500 - Train Loss: 1.4815, Train Acc: 34.92% - Val loss: 1.1544, Val Acc: 39.29%
2025-09-03 18:22:23,672 - Early stopping à l'epoch 177
2025-09-03 18:22:23,673 - Fold 8 - Accuracy finale: 57.14%
2025-09-03 18:22:23,674 - === Fold 9/10 ===
2025-09-03 18:22:23,692 - Epoch 1/500 - Train Loss: 1.5014, Train Acc: 33.33% - Val loss: 1.2500, Val Acc: 39.29%
2025-09-03 18:22:25,924 - Early stopping à l'epoch 147
2025-09-03 18:22:25,925 - Fold 9 - Accuracy finale: 60.71%
2025-09-03 18:22:25,925 - === Fold 10/10 ===
2025-09-03 18:22:25,944 - Epoch 1/500 - Train Loss: 1.3996, Train Acc: 37.70% - Val loss: 1.0901, Val Acc: 46.43%
2025-09-03 18:22:28,560 - Early stopping à l'epoch 171
2025-09-03 18:22:28,561 - Fold 10 - Accuracy finale: 60.71%
2025-09-03 18:22:28,561 - 
2025-09-03 18:22:28,561 - Résultats de la validation croisée
2025-09-03 18:22:28,561 - 
2025-09-03 18:22:28,561 - Accuracy moyenne: 61.79% ± 8.76%
2025-09-03 18:22:28,561 - Loss moyenne: 0.8031 ± 0.1460
2025-09-03 18:22:28,561 - Nombre d'epochs moyen: 138
2025-09-03 18:22:28,561 - Accuracy par fold: ['82.14%', '67.86%', '64.29%', '57.14%', '46.43%', '57.14%', '64.29%', '57.14%', '60.71%', '60.71%']

...

2025-09-03 18:32:11,790 - Variable candidate tBodyAcc-std()-X testée
...

2025-09-03 18:32:35,322 - Ajout de la variable tBodyGyro-mean()-X, mean accuracy - std accuracy: 80.4170

2025-09-03 18:32:35,327 - Variables actuellement selectionnées : ['tBodyGyro-mean()-X']

2025-09-03 18:32:35,328 - Variable candidate fBodyAcc-bandsEnergy()-1,16 testée
...

...

2025-09-03 19:37:59,040 - Plus d'amélioration -> arret

2025-09-03 19:37:59,040 - Variables sélectionnées : ['tBodyGyro-mean()-X', 'angle(tBodyGyroMean,gravityMean)', 'fBodyAccMag-mad()', 'tBodyAcc-mad()-X', 'fBodyAcc-bandsEnergy()-9,16', 'fBodyAccJerk-bandsEnergy()-1,16', 'fBodyAcc-bandsEnergy()-1,16']
2025-09-03 19:37:59,040 - 
2025-09-03 19:37:59,040 - Résultats de la validation croisée pour le jeu d'hyperparamètres 1
2025-09-03 19:37:59,040 - 
2025-09-03 19:37:59,040 - Accuracy moyenne: 95.36% ± 2.29%
2025-09-03 19:37:59,040 - Loss moyenne: 0.2621 ± 0.1640
2025-09-03 19:37:59,040 - Nombre d'epochs moyen: 176
2025-09-03 19:37:59,040 - Accuracy par fold: ['96.43%', '92.86%', '96.43%', '96.43%', '92.86%', '96.43%', '96.43%', '100.00%', '92.86%', '92.86%']

2025-09-03 19:37:59,041 - Entrainement du modèle final pour le jeu d'hyperparamètres 1

2025-09-03 19:37:59,061 - Epoch 1/176 - Loss: 1.3315, Acc: 38.93%
2025-09-03 19:38:01,888 - Epoch 176/176 - Loss: 0.0095, Acc: 99.64%
2025-09-03 19:38:01,890 - 

2025-09-03 19:38:01,890 - Résultats finaux pour le jeu d'hyperparamètres 1

2025-09-03 19:38:01,890 - Accuracy de validation croisée: 95.36% ± 2.29%
2025-09-03 19:38:01,890 - Accuracy sur le test: 95.00%
```

<br>

<p align="center">
<b>Results obtained during the search for the best estimator of the optimal classifier using the 6 main variables (see /logs) :</b><br>
</p>

<br>

|   index_set |   mean_acc |   std_acc |   test_accuracy | estimator_name   |   num_layers | hidden_size   | d_model   | nhead   | out_channels   |   dropout |   weight_decay |     lr |   scheduler_factor |   batch_size |   score |
|------------:|-----------:|----------:|----------------:|:-----------------|-------------:|:--------------|:----------|:--------|:---------------|----------:|---------------:|-------:|-------------------:|-------------:|--------:|
|          25 |       95.4 |       3.2 |            93.3 | Transformer      |            2 | -             | 36.0      | 4.0     | -              |     0.05  |        1e-06   | 0.0055 |                0.6 |           28 |    92.1 |
|          88 |       94.3 |       2.4 |            85   | Transformer      |            2 | -             | 36.0      | 4.0     | -              |     0.125 |        7.5e-07 | 0.005  |                0.6 |           32 |    91.9 |
|          67 |       93.9 |       4.8 |            80.8 | Transformer      |            2 | -             | 60.0      | 3.0     | -              |     0.15  |        1e-06   | 0.0045 |                0.5 |           32 |    89.1 |
|          78 |       93.2 |       4.4 |            85.8 | Transformer      |            2 | -             | 60.0      | 3.0     | -              |     0.075 |        7.5e-07 | 0.004  |                0.4 |           36 |    88.9 |
|          35 |       93.2 |       4.6 |            83.3 | Transformer      |            2 | -             | 36.0      | 2.0     | -              |     0.1   |        1e-06   | 0.0055 |                0.5 |           32 |    88.6 |
|          27 |       91.8 |       3.6 |            90   | Transformer      |            2 | -             | 48.0      | 12.0    | -              |     0     |        7.5e-07 | 0.0035 |                0.6 |           36 |    88.2 |
|          97 |       91.4 |       4   |            89.2 | Transformer      |            2 | -             | 60.0      | 2.0     | -              |     0     |        7.5e-07 | 0.0055 |                0.4 |           36 |    87.5 |
|          82 |       91.4 |       4   |            86.7 | Transformer      |            2 | -             | 60.0      | 3.0     | -              |     0     |        5e-07   | 0.0045 |                0.6 |           28 |    87.5 |
|          37 |       91.8 |       4.5 |            84.2 | Transformer      |            2 | -             | 72.0      | 3.0     | -              |     0     |        2.5e-06 | 0.0045 |                0.5 |           36 |    87.3 |
|          16 |       89.3 |       5.3 |            85   | Transformer      |            2 | -             | 72.0      | 4.0     | -              |     0.15  |        1e-06   | 0.005  |                0.4 |           32 |    84   |
|          75 |       85   |       8.4 |            75   | Transformer      |            2 | -             | 72.0      | 2.0     | -              |     0.075 |        5e-06   | 0.0055 |                0.6 |           36 |    76.6 |
|          80 |       84.3 |      18.6 |            82.5 | Transformer      |            3 | -             | 60.0      | 12.0    | -              |     0.175 |        5e-07   | 0.0035 |                0.5 |           36 |    65.7 |
|          73 |       73.9 |      25.8 |            59.2 | Transformer      |            4 | -             | 36.0      | 12.0    | -              |     0.025 |        2.5e-06 | 0.004  |                0.4 |           28 |    48.2 |
|          47 |       46.8 |      15   |            53.3 | Transformer      |            3 | -             | 60.0      | 4.0     | -              |     0.075 |        5e-06   | 0.0045 |                0.6 |           28 |    31.8 |
|           6 |       50.7 |      19.4 |            40   | Transformer      |            3 | -             | 60.0      | 12.0    | -              |     0.175 |        5e-06   | 0.005  |                0.5 |           28 |    31.3 |
|          81 |       55   |      25.8 |            43.3 | Transformer      |            3 | -             | 48.0      | 4.0     | -              |     0.1   |        5e-06   | 0.0055 |                0.6 |           36 |    29.2 |
|          33 |       40   |      17.8 |            19.2 | Transformer      |            4 | -             | 36.0      | 12.0    | -              |     0.1   |        1e-06   | 0.0045 |                0.4 |           28 |    22.2 |
|          55 |       28.6 |       8   |            37.5 | Transformer      |            3 | -             | 72.0      | 3.0     | -              |     0.1   |        5e-07   | 0.0045 |                0.4 |           36 |    20.6 |
|          17 |       33.6 |      14.9 |            47.5 | Transformer      |            3 | -             | 72.0      | 3.0     | -              |     0.1   |        7.5e-07 | 0.0045 |                0.4 |           28 |    18.7 |
|           1 |       30.4 |      11.9 |            20.8 | Transformer      |            3 | -             | 60.0      | 3.0     | -              |     0.05  |        5e-07   | 0.0055 |                0.6 |           36 |    18.5 |
|          36 |       20   |       1.7 |            20.8 | LSTM             |            3 | 80.0          | -         | -       | -              |     0.075 |        2.5e-06 | 0.0035 |                0.5 |           32 |    18.3 |
|          26 |       20   |       1.7 |            20.8 | CNN + LSTM       |            2 | 80.0          | -         | -       | 32.0           |     0     |        7.5e-07 | 0.0055 |                0.6 |           32 |    18.3 |
|          29 |       20   |       1.7 |            20.8 | CNN + LSTM       |            3 | 96.0          | -         | -       | 16.0           |     0.175 |        5e-06   | 0.0035 |                0.5 |           32 |    18.3 |
|          31 |       20   |       1.7 |            20.8 | CNN + LSTM       |            4 | 64.0          | -         | -       | 8.0            |     0     |        1e-06   | 0.0045 |                0.6 |           32 |    18.3 |
|          28 |       20   |       1.7 |            20.8 | CNN + LSTM       |            3 | 80.0          | -         | -       | 16.0           |     0.1   |        2.5e-06 | 0.004  |                0.5 |           36 |    18.3 |
|           3 |       20   |       1.7 |            20.8 | LSTM             |            2 | 80.0          | -         | -       | -              |     0     |        2.5e-06 | 0.005  |                0.4 |           32 |    18.3 |
|           2 |       20   |       1.7 |            20.8 | LSTM             |            3 | 96.0          | -         | -       | -              |     0.05  |        5e-06   | 0.0055 |                0.4 |           36 |    18.3 |
|           7 |       20   |       1.7 |            20.8 | CNN + LSTM       |            3 | 64.0          | -         | -       | 32.0           |     0.025 |        2.5e-06 | 0.005  |                0.6 |           32 |    18.3 |
|          15 |       20   |       1.7 |            20.8 | CNN + LSTM       |            3 | 80.0          | -         | -       | 32.0           |     0.075 |        5e-07   | 0.004  |                0.5 |           36 |    18.3 |
|          19 |       20   |       1.7 |            20.8 | CNN + LSTM       |            2 | 80.0          | -         | -       | 8.0            |     0.175 |        5e-06   | 0.004  |                0.6 |           32 |    18.3 |
|          23 |       20   |       1.7 |            20.8 | CNN + LSTM       |            3 | 96.0          | -         | -       | 16.0           |     0.1   |        2.5e-06 | 0.0045 |                0.4 |           36 |    18.3 |
|          24 |       20   |       1.7 |            20.8 | LSTM             |            4 | 64.0          | -         | -       | -              |     0.125 |        7.5e-07 | 0.0045 |                0.5 |           36 |    18.3 |
|          13 |       20   |       1.7 |            20.8 | LSTM             |            2 | 64.0          | -         | -       | -              |     0     |        7.5e-07 | 0.0045 |                0.5 |           36 |    18.3 |
|          14 |       20   |       1.7 |            20.8 | CNN + LSTM       |            3 | 64.0          | -         | -       | 16.0           |     0.175 |        5e-06   | 0.0045 |                0.6 |           32 |    18.3 |
|          12 |       20   |       1.7 |            20.8 | CNN + LSTM       |            4 | 64.0          | -         | -       | 32.0           |     0.1   |        5e-06   | 0.005  |                0.4 |           32 |    18.3 |
|          10 |       20   |       1.7 |            20.8 | LSTM             |            3 | 80.0          | -         | -       | -              |     0.075 |        5e-06   | 0.0035 |                0.5 |           28 |    18.3 |
|          92 |       20   |       1.7 |            20.8 | LSTM             |            3 | 96.0          | -         | -       | -              |     0.125 |        5e-06   | 0.0045 |                0.6 |           32 |    18.3 |
|          93 |       20   |       1.7 |            20.8 | LSTM             |            2 | 96.0          | -         | -       | -              |     0.125 |        7.5e-07 | 0.0045 |                0.5 |           32 |    18.3 |
|          96 |       20   |       1.7 |            20.8 | CNN + LSTM       |            3 | 80.0          | -         | -       | 16.0           |     0.025 |        7.5e-07 | 0.0035 |                0.5 |           28 |    18.3 |
|          95 |       20   |       1.7 |            20.8 | CNN + LSTM       |            2 | 96.0          | -         | -       | 32.0           |     0.175 |        5e-06   | 0.0055 |                0.4 |           36 |    18.3 |
|          89 |       20   |       1.7 |            20.8 | CNN + LSTM       |            4 | 64.0          | -         | -       | 8.0            |     0.1   |        5e-06   | 0.0035 |                0.4 |           36 |    18.3 |
|          68 |       20   |       1.7 |            20.8 | CNN + LSTM       |            3 | 96.0          | -         | -       | 16.0           |     0     |        2.5e-06 | 0.005  |                0.5 |           32 |    18.3 |
|          65 |       20   |       1.7 |            20.8 | LSTM             |            4 | 96.0          | -         | -       | -              |     0.025 |        5e-07   | 0.004  |                0.4 |           32 |    18.3 |
|          66 |       20   |       1.7 |            20.8 | CNN + LSTM       |            3 | 80.0          | -         | -       | 32.0           |     0.025 |        7.5e-07 | 0.0035 |                0.6 |           28 |    18.3 |
|          84 |       20   |       1.7 |            20.8 | LSTM             |            3 | 64.0          | -         | -       | -              |     0.125 |        1e-06   | 0.0035 |                0.6 |           32 |    18.3 |
|          79 |       20   |       1.7 |            20.8 | LSTM             |            3 | 96.0          | -         | -       | -              |     0.025 |        1e-06   | 0.004  |                0.6 |           32 |    18.3 |
|          22 |       20   |       1.7 |            20.8 | LSTM             |            2 | 96.0          | -         | -       | -              |     0.125 |        7.5e-07 | 0.004  |                0.5 |           36 |    18.3 |
|          21 |       20   |       1.7 |            20.8 | LSTM             |            4 | 96.0          | -         | -       | -              |     0.025 |        2.5e-06 | 0.004  |                0.4 |           32 |    18.3 |
|          43 |       20   |       1.7 |            20.8 | LSTM             |            3 | 96.0          | -         | -       | -              |     0.1   |        5e-06   | 0.0055 |                0.5 |           32 |    18.3 |
|          46 |       20   |       1.7 |            20.8 | CNN + LSTM       |            4 | 96.0          | -         | -       | 32.0           |     0.025 |        2.5e-06 | 0.0035 |                0.4 |           32 |    18.3 |
|          40 |       20   |       1.7 |            20.8 | CNN + LSTM       |            2 | 96.0          | -         | -       | 8.0            |     0.025 |        1e-06   | 0.005  |                0.5 |           32 |    18.3 |
|          38 |       20   |       1.7 |            20.8 | LSTM             |            2 | 64.0          | -         | -       | -              |     0.05  |        7.5e-07 | 0.0055 |                0.6 |           36 |    18.3 |
|          64 |       20   |       1.7 |            20.8 | LSTM             |            2 | 80.0          | -         | -       | -              |     0.15  |        5e-06   | 0.005  |                0.6 |           28 |    18.3 |
|          61 |       20   |       1.7 |            20.8 | LSTM             |            4 | 80.0          | -         | -       | -              |     0.025 |        2.5e-06 | 0.005  |                0.4 |           32 |    18.3 |
|          63 |       20   |       1.7 |            20.8 | LSTM             |            4 | 96.0          | -         | -       | -              |     0.175 |        2.5e-06 | 0.005  |                0.6 |           32 |    18.3 |
|          62 |       20   |       1.7 |            20.8 | CNN + LSTM       |            2 | 64.0          | -         | -       | 16.0           |     0     |        7.5e-07 | 0.004  |                0.5 |           36 |    18.3 |
|          57 |       20   |       1.7 |            20.8 | LSTM             |            4 | 64.0          | -         | -       | -              |     0.025 |        5e-06   | 0.004  |                0.5 |           28 |    18.3 |
|          59 |       20   |       1.7 |            20.8 | LSTM             |            3 | 64.0          | -         | -       | -              |     0.175 |        5e-07   | 0.0055 |                0.4 |           32 |    18.3 |
|          60 |       20   |       1.7 |            20.8 | LSTM             |            3 | 96.0          | -         | -       | -              |     0.175 |        2.5e-06 | 0.005  |                0.5 |           32 |    18.3 |
|          56 |       20   |       1.7 |            20.8 | CNN + LSTM       |            4 | 64.0          | -         | -       | 32.0           |     0.025 |        2.5e-06 | 0.0055 |                0.5 |           28 |    18.3 |
|          51 |       20   |       1.7 |            20.8 | CNN + LSTM       |            4 | 96.0          | -         | -       | 32.0           |     0.1   |        7.5e-07 | 0.0045 |                0.5 |           28 |    18.3 |
|          50 |       20   |       1.7 |            20.8 | CNN + LSTM       |            2 | 64.0          | -         | -       | 32.0           |     0.125 |        5e-07   | 0.004  |                0.5 |           32 |    18.3 |
|          52 |       20   |       1.7 |            20.8 | CNN + LSTM       |            4 | 96.0          | -         | -       | 16.0           |     0.05  |        5e-06   | 0.0035 |                0.5 |           32 |    18.3 |
|          49 |       20   |       1.7 |            20.8 | CNN + LSTM       |            2 | 64.0          | -         | -       | 32.0           |     0.1   |        5e-06   | 0.004  |                0.6 |           32 |    18.3 |
|          53 |       20   |       1.7 |            20.8 | CNN + LSTM       |            2 | 80.0          | -         | -       | 8.0            |     0.15  |        2.5e-06 | 0.005  |                0.6 |           36 |    18.3 |
|          54 |       20   |       1.7 |            20.8 | LSTM             |            3 | 64.0          | -         | -       | -              |     0.125 |        1e-06   | 0.0045 |                0.4 |           28 |    18.3 |
|          45 |       20   |       1.7 |            20.8 | CNN + LSTM       |            4 | 80.0          | -         | -       | 32.0           |     0.1   |        7.5e-07 | 0.004  |                0.4 |           32 |    18.3 |
|          41 |       20   |       1.7 |            20.8 | LSTM             |            2 | 80.0          | -         | -       | -              |     0.075 |        7.5e-07 | 0.0055 |                0.5 |           36 |    18.3 |
|          99 |       20   |       1.7 |            20.8 | LSTM             |            4 | 64.0          | -         | -       | -              |     0     |        7.5e-07 | 0.004  |                0.5 |           32 |    18.3 |
|          98 |       20   |       1.7 |            20.8 | LSTM             |            3 | 64.0          | -         | -       | -              |     0.075 |        5e-07   | 0.004  |                0.4 |           28 |    18.3 |
|         100 |       20   |       1.7 |            20.8 | CNN + LSTM       |            4 | 80.0          | -         | -       | 16.0           |     0.025 |        7.5e-07 | 0.005  |                0.4 |           32 |    18.3 |
|          87 |       20   |       1.7 |            20.8 | LSTM             |            2 | 64.0          | -         | -       | -              |     0.175 |        1e-06   | 0.005  |                0.5 |           36 |    18.3 |
|          91 |       20   |       1.7 |            20.8 | LSTM             |            2 | 96.0          | -         | -       | -              |     0.175 |        2.5e-06 | 0.0035 |                0.4 |           32 |    18.3 |
|          90 |       20   |       1.7 |            20.8 | LSTM             |            3 | 64.0          | -         | -       | -              |     0.1   |        7.5e-07 | 0.005  |                0.5 |           28 |    18.3 |
|          70 |       20   |       1.7 |            20.8 | CNN + LSTM       |            3 | 96.0          | -         | -       | 16.0           |     0.025 |        7.5e-07 | 0.004  |                0.6 |           32 |    18.3 |
|          71 |       20   |       1.7 |            20.8 | LSTM             |            4 | 96.0          | -         | -       | -              |     0     |        2.5e-06 | 0.0035 |                0.6 |           28 |    18.3 |
|          85 |       20   |       1.7 |            20.8 | LSTM             |            2 | 96.0          | -         | -       | -              |     0.025 |        5e-07   | 0.0035 |                0.6 |           32 |    18.3 |
|          86 |       20   |       1.7 |            20.8 | CNN + LSTM       |            4 | 80.0          | -         | -       | 8.0            |     0     |        1e-06   | 0.0055 |                0.6 |           32 |    18.3 |
|          72 |       20   |       1.7 |            20.8 | LSTM             |            4 | 80.0          | -         | -       | -              |     0.1   |        5e-06   | 0.004  |                0.6 |           28 |    18.3 |
|          74 |       20   |       1.7 |            20.8 | CNN + LSTM       |            3 | 64.0          | -         | -       | 16.0           |     0.15  |        5e-07   | 0.0035 |                0.4 |           28 |    18.3 |
|          11 |       20.7 |       2.7 |            20.8 | Transformer      |            4 | -             | 72.0      | 4.0     | -              |     0     |        5e-06   | 0.005  |                0.5 |           28 |    18   |
|          83 |       19.6 |       1.8 |            20.8 | Transformer      |            4 | -             | 72.0      | 4.0     | -              |     0.15  |        5e-06   | 0.0055 |                0.6 |           32 |    17.9 |
|          76 |       19.3 |       1.7 |            20.8 | CNN + LSTM       |            2 | 64.0          | -         | -       | 16.0           |     0.075 |        5e-06   | 0.0035 |                0.5 |           32 |    17.5 |
|           5 |       18.9 |       1.6 |            20.8 | CNN + LSTM       |            2 | 96.0          | -         | -       | 8.0            |     0.15  |        5e-07   | 0.0045 |                0.4 |           36 |    17.3 |
|          42 |       18.9 |       1.6 |            19.2 | CNN + LSTM       |            2 | 96.0          | -         | -       | 8.0            |     0.125 |        5e-07   | 0.0035 |                0.5 |           36 |    17.3 |
|          20 |       18.9 |       1.6 |            19.2 | Transformer      |            4 | -             | 72.0      | 12.0    | -              |     0.15  |        2.5e-06 | 0.0045 |                0.6 |           36 |    17.3 |
|          32 |       18.9 |       1.6 |            20.8 | CNN + LSTM       |            2 | 96.0          | -         | -       | 8.0            |     0.15  |        5e-07   | 0.0055 |                0.6 |           28 |    17.3 |
|          39 |       18.6 |       1.4 |            20.8 | CNN + LSTM       |            2 | 96.0          | -         | -       | 16.0           |     0.1   |        5e-07   | 0.0035 |                0.4 |           32 |    17.1 |
|          18 |       22.1 |       5.5 |            19.2 | Transformer      |            4 | -             | 48.0      | 4.0     | -              |     0.025 |        5e-06   | 0.005  |                0.6 |           28 |    16.7 |
|          94 |       24.6 |       8.8 |            20.8 | Transformer      |            4 | -             | 60.0      | 2.0     | -              |     0.025 |        5e-07   | 0.004  |                0.5 |           28 |    15.8 |
|          30 |       20.7 |       5.5 |            20.8 | Transformer      |            4 | -             | 72.0      | 4.0     | -              |     0.05  |        2.5e-06 | 0.005  |                0.5 |           36 |    15.2 |
|           4 |       21.8 |       7.2 |            20.8 | Transformer      |            4 | -             | 36.0      | 2.0     | -              |     0.075 |        1e-06   | 0.0055 |                0.5 |           28 |    14.6 |
|          44 |       21.4 |       7.5 |            20.8 | Transformer      |            4 | -             | 48.0      | 2.0     | -              |     0.175 |        7.5e-07 | 0.005  |                0.6 |           28 |    13.9 |
|           8 |       22.9 |       9.2 |            20.8 | Transformer      |            4 | -             | 48.0      | 12.0    | -              |     0.05  |        7.5e-07 | 0.0055 |                0.5 |           36 |    13.7 |
|          48 |       21.8 |       8.4 |            48.3 | Transformer      |            4 | -             | 60.0      | 12.0    | -              |     0.125 |        2.5e-06 | 0.005  |                0.6 |           32 |    13.4 |
|          58 |       24.3 |      11   |            19.2 | Transformer      |            4 | -             | 36.0      | 12.0    | -              |     0.15  |        7.5e-07 | 0.005  |                0.6 |           32 |    13.2 |
|          34 |       36.4 |      23.3 |            20.8 | Transformer      |            4 | -             | 48.0      | 12.0    | -              |     0.15  |        1e-06   | 0.0035 |                0.5 |           28 |    13.1 |
|           9 |       24.3 |      11.2 |            41.7 | Transformer      |            4 | -             | 60.0      | 3.0     | -              |     0     |        2.5e-06 | 0.004  |                0.6 |           36 |    13.1 |
|          77 |       33.2 |      21.7 |            28.3 | Transformer      |            4 | -             | 36.0      | 12.0    | -              |     0.075 |        2.5e-06 | 0.005  |                0.5 |           36 |    11.5 |
|          69 |       32.5 |      24.1 |            20.8 | Transformer      |            4 | -             | 36.0      | 4.0     | -              |     0.15  |        7.5e-07 | 0.0055 |                0.5 |           36 |     8.4 |

