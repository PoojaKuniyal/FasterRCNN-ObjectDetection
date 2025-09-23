import torch
from torch.optim import Adam
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm
from src.logger import get_logger
import torchvision
from src.custom_exception import CustomException
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

logger = get_logger(__name__)

class FasterRCNNModel:
    def __init__(self, num_classes, device):
        self.num_classes = num_classes
        self.device = device
        self.optimizer = None
        self.model = self.create_model().to(self.device)

        logger.info("Model Architecture initialized....")


    def create_model(self):
        
        try:
            try:
                weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
                model = fasterrcnn_resnet50_fpn(weights=weights)
            except Exception:
                model = fasterrcnn_resnet50_fpn(pretrained=True)

            # Replace the classification head with correct number of classes
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
            return model
        except Exception as e:
            logger.error(f"Failed to create model {e}.")
            raise CustomException("Failed to create model",e)
    


    