import torch.utils.data as data
from PIL import Image
import numpy as np
import pandas as pd
import os

class CombinedDataset(data.Dataset):
    def __init__(self,df, img_dir, transforms):
        self.img_dir = img_dir
        self.transforms = transforms
        self.df = pd.read_excel(df)
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        current_name = self.df.iloc[idx, 0]
        # 确保 current_name 是相对路径
        if os.path.isabs(current_name):
            # 去掉开头的斜杠，使其成为相对路径
            current_name = current_name.lstrip('/')
        path = os.path.join(self.img_dir, current_name)
        # image = Image.open(path).convert("L")
        image = Image.open(path).convert("RGB")

        image = np.array(image)

        image = Image.fromarray(image)
        image = self.transforms(image)

        bcva=self.df.iloc[idx,1]

        cst = self.df.iloc[idx,2]
        eye_id = self.df.iloc[idx, 3]
        patient = self.df.iloc[idx,4]

        return image, bcva,cst,eye_id,patient
if __name__ == '__main__':
    df_path = "/home/gu721/yzc/data/ophthalmic_multimodal/OLIVES_Dataset_Labels/ml_centric_labels/Clinical_Data_Images.xlsx"
    img_dir = "/home/gu721/yzc/data/ophthalmic_multimodal/OLIVES"
    dataset = CombinedDataset(df_path,img_dir,transforms = None)
    print(dataset[0])
