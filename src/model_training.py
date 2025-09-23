import os
import torch
from torch.utils.data import DataLoader, random_split
from torch import optim
from src.model_architecture import FasterRCNNModel
from src.logger import get_logger
from src.custom_exception import CustomException
from src.data_processing import GunDataset
from config.paths_config import *

from torch.utils.tensorboard import SummaryWriter
import time

logger = get_logger(__name__)

model_save_path = MODEL_SAVE_PATH
os.makedirs(model_save_path, exist_ok=True)

class ModelTraining:
    def __init__(self, model_class, num_classes, learning_rate, epochs, dataset_path, device):
        self.model_class = model_class
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.dataset_path = dataset_path
        self.device = device
        
        ##### TENSORBOARD
        ##      log directory for tensorboard
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.log_dir = f"tensorboard_logs/{timestamp}" # RUNS (running model training file 1,2..) will get stored in this format)
        os.makedirs(self.log_dir, exist_ok=True)

        # SummaryWriter: logging to tensorboard
        self.writer = SummaryWriter(log_dir=self.log_dir)

        try:
            # initialize model
            self.model_wrapper = self.model_class(self.num_classes, self.device)
            self.model = self.model_wrapper.model
            self.model.to(self.device)
            logger.info("Model moved to device")

            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
           
            logger.info("Optimizer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize model training {e}")
            raise CustomException("Failed to initialize model training", e)

    def collate_fn(self, batch):
        return tuple(zip(*batch))

    def split_dataset(self):
        try:
            dataset = GunDataset(self.dataset_path, self.device)

            dataset = torch.utils.data.Subset(dataset, range(300))

            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

            train_loader = DataLoader(
                train_dataset, batch_size=3, shuffle=True, num_workers=0, collate_fn=self.collate_fn
            )
            val_loader = DataLoader(
                val_dataset, batch_size=3, shuffle=False, num_workers=0, collate_fn=self.collate_fn
            )

            logger.info("Dataset split successfully")
            return train_loader, val_loader

        except Exception as e:
            logger.error(f"Failed to split dataset {e}")
            raise CustomException("Failed to split dataset", e)
        

    def train(self):
        
        try:
            train_loader, val_loader = self.split_dataset()

            for epoch in range(self.epochs):
                logger.info(f"Starting epoch {epoch+1}/{self.epochs}")

                # ---------------- TRAINING ----------------
                self.model.train()
                running_train_loss = 0.0

                for i, (images, targets) in enumerate(train_loader):
                    images = [img.to(self.device) for img in images]
                    targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                    self.optimizer.zero_grad()
                    loss_dict = self.model(images, targets)
                    total_loss = sum(loss for loss in loss_dict.values())

                    if total_loss.item() == 0:
                        logger.error("Zero training loss detected")
                        raise ValueError("Total training loss is zero")

                    # log train loss per batch
                    self.writer.add_scalar("Train Loss ", total_loss.item(), epoch*len(train_loader)+i)
                    
                    total_loss.backward()
                    self.optimizer.step()

                    running_train_loss += total_loss.item()

                    # Log running total (per batch)
                    self.writer.add_scalar("Running Train Loss", running_train_loss, epoch*len(train_loader)+i)
                
                avg_train_loss = running_train_loss / max(1, len(train_loader))
                # Log average loss (per epoch)
                self.writer.add_scalar("Average Train Loss per Epoch", avg_train_loss, epoch)

                logger.info(f"Epoch {epoch+1} - Avg Train Loss: {avg_train_loss:.4f}")

                self.writer.flush()
                
                # ---------------- VALIDATION ----------------
                self.model.train()  # keep train() mode for loss computation
                running_val_loss = 0.0

                with torch.no_grad():
                    for images, targets in val_loader:
                        images = [img.to(self.device) for img in images]
                        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                        loss_dict = self.model(images, targets)
                        total_val_loss = sum(loss for loss in loss_dict.values())
                        
                        self.writer.add_scalar("Train Loss ", total_loss.item(), epoch*len(train_loader)+i)
                        
                        running_val_loss += total_val_loss.item()

                avg_val_loss = running_val_loss / max(1, len(val_loader))
                self.writer.add_scalar("Average Val Loss per Epoch", avg_val_loss, epoch)
                logger.info(f"Epoch {epoch+1} - Avg Val Loss: {avg_val_loss:.4f}")

                self.writer.flush()


                # ---------------- SAVE MODEL ----------------
                model_path = os.path.join(model_save_path, "fasterrcnn.pth")
                torch.save(self.model.state_dict(), model_path)
                logger.info(f"Model saved at {model_path}")

            logger.info("Training completed successfully")

        except Exception as e:
            logger.error(f"Failed to train model {e}")
            raise CustomException("Failed to train model", e)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    training = ModelTraining(
        model_class=FasterRCNNModel,
        num_classes=2,
        learning_rate=0.0001,
        dataset_path=ROOT_PATH,
        device=device,
        epochs=1,
          )
    training.train()

 # cmd command for checking logs: tensorboard --log_dir=tensorboard_logs/