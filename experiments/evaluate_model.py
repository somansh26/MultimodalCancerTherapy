import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from models.fusion_model import TransformerFusionModel
from preprocessing.preprocess_data import preprocess_imaging, preprocess_text, preprocess_omics
from utils.logger import setup_logger


# Simulated Dataset Class
class MultimodalTestDataset(Dataset):
    def __init__(self, num_samples=200):
        self.images = torch.rand((num_samples, 3, 224, 224))  
        self.texts = ["Radiology report example"] * num_samples  
        self.omics = torch.rand((num_samples, 1000)) 
        self.targets = torch.randint(0, 2, (num_samples, 1)) 

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return {
            "image": self.images[idx],
            "text": self.texts[idx],
            "omics": self.omics[idx],
            "target": self.targets[idx],
        }


# Function to Compute Metrics
def compute_metrics(y_true, y_pred_logits):
    """
    Compute evaluation metrics like ROC-AUC and F1-Score.
    """
    y_pred = torch.sigmoid(y_pred_logits).detach().cpu().numpy()
    y_pred_binary = (y_pred > 0.5).astype(int)

    roc_auc = roc_auc_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred_binary)

    cm = confusion_matrix(y_true, y_pred_binary)
    return {"ROC-AUC": roc_auc, "F1-Score": f1, "Confusion Matrix": cm}


# Function to Plot ROC Curve
def plot_roc_curve(y_true, y_pred_logits):
    """
    Plot the Receiver Operating Characteristic (ROC) curve.
    """
    y_pred = torch.sigmoid(y_pred_logits).detach().cpu().numpy()
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color="blue", label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig("roc_curve.png")
    plt.show()


# Evaluation Function
def evaluate_model(model, dataloader, device, logger):
    """
    Evaluate the model on the test dataset.
    """
    model.eval()
    y_true = []
    y_pred_logits = []

    with torch.no_grad():
        for batch in dataloader:
            # Move data to device
            image = batch["image"].to(device)
            text = preprocess_text(batch["text"]).to(device)
            omics = batch["omics"].to(device)
            target = batch["target"].to(device)

            # Predict
            output_logits = model(image, text, omics)
            y_true.extend(target.cpu().numpy())
            y_pred_logits.extend(output_logits.cpu().numpy())

    y_true = torch.tensor(y_true).flatten().numpy()
    y_pred_logits = torch.tensor(y_pred_logits).flatten()

    metrics = compute_metrics(y_true, y_pred_logits)
    logger.info(f"Evaluation Metrics: ROC-AUC: {metrics['ROC-AUC']:.4f}, F1-Score: {metrics['F1-Score']:.4f}")
    logger.info(f"Confusion Matrix:\n{metrics['Confusion Matrix']}")

    # Plot ROC Curve
    plot_roc_curve(y_true, y_pred_logits)

    return metrics


# Main Script
if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logger()

    model = TransformerFusionModel(omics_input_dim=1000).to(device)
    model.load_state_dict(torch.load("checkpoints/fusion_model.pth"))
    logger.info("Model loaded.")

    test_dataset = MultimodalTestDataset(num_samples=200)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    logger.info("Starting evaluation...")
    metrics = evaluate_model(model, test_dataloader, device, logger)
    logger.info(f"Final Metrics: {metrics}")
