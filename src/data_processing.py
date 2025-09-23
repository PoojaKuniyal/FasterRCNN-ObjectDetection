import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from src.logger import get_logger
from config.paths_config import *
from src.custom_exception import CustomException

logger = get_logger(__name__)

class GunDataset(Dataset):

    # Pass device as an argument to the class
    def __init__(self,root:str, device:str ="cpu"):
        self.image_path = os.path.join(root,"Images")
        self.labels_path = os.path.join(root,"Labels")
        self.device = device

        self.img_name =  sorted(os.listdir(self.image_path))
        self.label_name = sorted(os.listdir(self.labels_path))

        logger.info("Data Processing Initialized.... ")


    def __getitem__(self,idx):
        try:
            logger.info(f"Loading data for index {idx}")

            ##### LOADING IMAGES #######
            image_path =  os.path.join(self.image_path , str(self.img_name[idx]))
            
            logger.info(f'Image path : {image_path}')
            image = cv2.imread(image_path)
            img_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB).astype(np.float32)

            img_res = img_rgb/255
            img_res = torch.as_tensor(img_res).permute(2,0,1)

            ######## LOADING LABELS ##########
            label_name  = self.img_name[idx].rsplit('.',1)[0]+ ".txt"
            label_path = os.path.join(self.labels_path , str(label_name))

            if not os.path.exists(label_path):
                raise FileNotFoundError(f"Label file not found : {label_path}")
            

            target ={
                "boxes" : torch.tensor([]),
                "area" : torch.tensor([]),
                "image_id": torch.tensor([idx]),
                "labels": torch.tensor([], dtype=torch.int64)
            }

            with open(label_path,"r") as label_file:
                l_count = int(label_file.readline())
                box=[list(map(int, label_file.readline().split())) for _ in range(l_count)]
                # for i in range(l_count):
                #     box.append(list(map(int,label_file.readline().split())))
                    

            if box:
                area = [(b[2]-b[0]) * (b[3]-b[1]) for b in box] 
                labels = [1] * len(box)
            

                target["boxes"] = torch.tensor(box, dtype= torch.float32)
                target["area"] =  torch.tensor(area, dtype= torch.float32)
                target["labels"] = torch.tensor(labels , dtype=torch.int64)

            img_res = img_res.to(self.device)
            for key in target:
                #if isinstance(target[key] , torch.Tensor): no need for this code already converted to tensor
                target[key] = target[key].to(self.device)

            return img_res,target
        
        except Exception as e:
            logger.error("Error while loading data {e}")
            raise CustomException("Failed to load data",e)


    def __len__(self):
        return len(self.img_name)
    
if __name__ == "__main__":
    root_path = ROOT_PATH
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = GunDataset(root=root_path, device=device)
    image, target = dataset[0]

    print("image shape ", image.shape)
    print("target keys ", target.keys())
    print("bounding boxes ",target["boxes"])