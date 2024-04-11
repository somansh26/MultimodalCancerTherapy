import os
import requests
import zipfile
import pandas as pd
import numpy as np
import cv2
from preprocessing.preprocess_data import preprocess_imaging, preprocess_text, preprocess_omics

def download_and_extract(url, output_dir):
    """
    Downloads and extracts a dataset from a given URL.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    zip_path = os.path.join(output_dir, "dataset.zip")

    # Download dataset
    response = requests.get(url, stream=True)
    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            f.write(chunk)
    print(f"Downloaded dataset to {zip_path}")

    # Extract dataset
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)
    print(f"Extracted dataset to {output_dir}")

    # Remove zip file
    os.remove(zip_path)


def load_image_dataset(image_dir, labels_csv, augment=False):
    """
    Loads and preprocesses an imaging dataset.
    Args:
        image_dir (str): Directory containing image files.
        labels_csv (str): Path to a CSV file with image filenames and labels.
        augment (bool): Whether to apply data augmentation.
    Returns:
        tuple: Preprocessed image tensors and labels.
    """
    labels_df = pd.read_csv(labels_csv)
    images = []
    labels = []

    for _, row in labels_df.iterrows():
        image_path = os.path.join(image_dir, row["filename"])
        image = cv2.imread(image_path)
        if image is not None:
            preprocessed_image = preprocess_imaging(image, augment=augment)
            images.append(preprocessed_image)
            labels.append(row["label"])

    return np.array(images), np.array(labels)


def load_text_dataset(text_csv):
    """
    Loads and preprocesses a text dataset.
    Args:
        text_csv (str): Path to a CSV file with text data and labels.
    Returns:
        tuple: Tokenized text data and labels.
    """
    text_df = pd.read_csv(text_csv)
    texts = text_df["text"].tolist()
    labels = text_df["label"].tolist()

    preprocessed_texts = preprocess_text(texts)
    return preprocessed_texts, np.array(labels)


def load_omics_dataset(omics_csv):
    """
    Loads and preprocesses an omics dataset.
    Args:
        omics_csv (str): Path to a CSV file with omics data and labels.
    Returns:
        tuple: Preprocessed omics data and labels.
    """
    omics_df = pd.read_csv(omics_csv)
    labels = omics_df["label"].tolist()
    omics_data = omics_df.drop(columns=["label"]).values

    preprocessed_omics = preprocess_omics(omics_data)
    return preprocessed_omics, np.array(labels)


def load_multimodal_dataset(image_dir, image_csv, text_csv, omics_csv):
    """
    Loads and preprocesses a multimodal dataset.
    Args:
        image_dir (str): Directory containing image files.
        image_csv (str): Path to CSV with image filenames and labels.
        text_csv (str): Path to CSV with text data and labels.
        omics_csv (str): Path to CSV with omics data and labels.
    Returns:
        dict: Dictionary with preprocessed data for each modality and labels.
    """
    images, image_labels = load_image_dataset(image_dir, image_csv, augment=True)
    texts, text_labels = load_text_dataset(text_csv)
    omics, omics_labels = load_omics_dataset(omics_csv)

    # Ensure all labels match
    assert np.array_equal(image_labels, text_labels)
    assert np.array_equal(image_labels, omics_labels)

    return {
        "images": images,
        "texts": texts,
        "omics": omics,
        "labels": image_labels,
    }
