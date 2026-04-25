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
