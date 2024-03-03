import numpy as np
import cv2
from transformers import BertTokenizer
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

def preprocess_imaging(image, augment=False):
    """
    Preprocess an image:
    - Normalize intensity
    - Resize to 224x224
    - Optionally augment the image
    """
    normalized_image = (image - np.mean(image)) / np.std(image)
    resized_image = cv2.resize(normalized_image, (224, 224))

    if augment:
        angle = np.random.randint(-15, 15)
        rotation_matrix = cv2.getRotationMatrix2D((112, 112), angle, 1)
        resized_image = cv2.warpAffine(resized_image, rotation_matrix, (224, 224))

    return resized_image

def preprocess_text(text, max_length=512):
    """
    Preprocess text using BERT tokenizer.
    """
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenized = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    return tokenized

def preprocess_omics(omics_df):
    """
    Preprocess omics data:
    - Handle missing values
    - Scale features
    """
    imputer = KNNImputer(n_neighbors=5)
    imputed_data = imputer.fit_transform(omics_df)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(imputed_data)

    top_features = scaled_data[:, :100]  # Select top 100 features
    return top_features

def preprocess_all(image, text, omics):
    """
    Unified preprocessing for all modalities.
    """
    processed_image = preprocess_imaging(image)
    processed_text = preprocess_text(text)
    processed_omics = preprocess_omics(omics)

    return processed_image, processed_text, processed_omics

















