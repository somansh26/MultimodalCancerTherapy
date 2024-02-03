import os
import torch
from torch.utils.data import DataLoader, Dataset
from models.fusion_model import TransformerFusionModel
from preprocessing.preprocess_data import preprocess_imaging, preprocess_text, preprocess_omics
from utils.logger import setup_logger
from utils.metrics import compute_metrics
import matplotlib.pyplot as plt


# Simulated Dataset Class
class MultimodalDataset(Dataset):
    def __init__(self, num_samples=1000):
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


# Training Function
def train_model(model, dataloader, criterion, optimizer, device, logger, num_epochs=5):
    model.train()
    train_losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for batch in dataloader:

            image = batch["image"].to(device)
            text = preprocess_text(batch["text"]).to(device)  
            omics = batch["omics"].to(device)
            target = batch["target"].to(device)

            # Forward pass
            optimizer.zero_grad()
            output = model(image, text, omics)
            loss = criterion(output, target.float())

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        train_losses.append(avg_loss)
        logger.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

    return train_losses


def plot_loss(train_losses):
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Loss over Epochs")
    plt.savefig("training_loss.png")
    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logger()

    batch_size = 32
    learning_rate = 1e-4
    num_epochs = 10

    model = TransformerFusionModel(omics_input_dim=1000).to(device)
    logger.info("Model initialized.")

    dataset = MultimodalDataset(num_samples=500)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    logger.info("Starting training...")
    train_losses = train_model(model, dataloader, criterion, optimizer, device, logger, num_epochs=num_epochs)

    plot_loss(train_losses)

    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
    torch.save(model.state_dict(), "checkpoints/fusion_model.pth")
    logger.info("Model saved.")
