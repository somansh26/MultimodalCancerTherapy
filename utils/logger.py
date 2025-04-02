import logging
import wandb

def setup_logger(log_file="experiment.log"):
    """
    Sets up a logger to log experiment details to both the console and a file.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()
    return logger


def initialize_wandb(project_name, config):
    """
    Initializes Weights & Biases (wandb) for experiment tracking.
    Args:
        project_name (str): Name of the wandb project.
        config (dict): Configuration dictionary with experiment details.
    """
    wandb.init(project=project_name, config=config)
    logging.info(f"Initialized Weights & Biases for project: {project_name}")


def log_metrics_wandb(metrics, step):
    """
    Logs metrics to Weights & Biases.
    Args:
        metrics (dict): Dictionary of metrics to log.
        step (int): Training step or epoch.
    """
    wandb.log(metrics, step=step)
    logging.info(f"Logged metrics to wandb: {metrics}")
