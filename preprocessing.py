import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import logging

logger = logging.getLogger(__name__)

def preprocess(csv_file_name: str, features: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    df = pd.read_csv(csv_file_name)

    df['Group'] = (df['Activity'] != df['Activity'].shift()).cumsum()

    grouped = df.groupby('Group')

    X = [group[features].values for _, group in grouped]
    y = [group['Activity'].values[0] for _, group in grouped]

    label_encoder = LabelEncoder()
    label_encoder.classes_ = ['LAYING', 'SITTING', 'STANDING', 'WALKING', 'WALKING_DOWNSTAIRS', 'WALKING_UPSTAIRS']
    y = label_encoder.fit_transform(y)

    y = torch.tensor(y, dtype=torch.long) # torch.Size([n_sequences])
    logger.info(f"Il y a {y.size(0)} séquences de mesures") # train: 280 & test: 120

    _, counts = torch.unique(y, return_counts=True)
    proportions = counts.float() / y.size(0)

    logger.info("Répartition des classes:")
    for label, count, prop in zip(label_encoder.classes_, counts, proportions):
        logger.info(f"{label.item()} : {count.item()} séquences ({prop.item()*100:.2f}%)")

    all_data = np.vstack(X)
    scaler = StandardScaler()
    scaler.fit(all_data)
    X = [scaler.transform(seq) for seq in X]

    X = [torch.tensor(seq, dtype=torch.float32) for seq in X]
    
    max_len = max([len(seq) for seq in X])
    logger.info(f"La plus longue séquence comporte {max_len} éléments\n") # train: 48 & test: 40

    X = [torch.cat([seq, torch.zeros(max_len - len(seq), len(features))]) if len(seq) < max_len else seq for seq in X]

    X = torch.stack(X) # torch.Size([n_sequences, n_timesteps, n_features=6])

    logger.info(f"Les données sont centrées-réduites et un padding de zeros est appliqué afin d'uniformiser la longueur des séquences à celle maximale\n")

    return X, y