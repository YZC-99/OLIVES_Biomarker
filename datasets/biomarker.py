import torch.utils.data as data
from PIL import Image
import numpy as np
import pandas as pd
import os

class BiomarkerDatasetAttributes(data.Dataset):
    def __init__(self,df, img_dir, transforms):
        self.img_dir = img_dir
        self.transforms = transforms
        self.df = pd.read_csv(df)
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        current_name = self.df.iloc[idx, 0]
        path = self.img_dir + current_name
        image = Image.open(path).convert("L")
        image = np.array(image)
        image = Image.fromarray(image)
        image = self.transforms(image)
        atrophy = self.df.iloc[idx,2]
        EZ = self.df.iloc[idx,3]
        DRIL = self.df.iloc[idx,4]
        IR_hemm = self.df.iloc[idx,5]
        ir_hrf = self.df.iloc[idx,6]
        partial_vit = self.df.iloc[idx,7]
        full_vit = self.df.iloc[idx,8]
        preret_tiss = self.df.iloc[idx,9]
        vit_deb = self.df.iloc[idx,10]
        vmt = self.df.iloc[idx,11]
        drt = self.df.iloc[idx,12]
        fluid_irf = self.df.iloc[idx,13]
        fluid_srf = self.df.iloc[idx,14]

        rpe = self.df.iloc[idx,15]
        ga = self.df.iloc[idx,16]
        shrm = self.df.iloc[idx,17]
        eye_id = self.df.iloc[idx,18]
        bcva = self.df.iloc[idx,19]
        cst = self.df.iloc[idx,20]
        patient = self.df.iloc[idx,21]

        # ga = self.df.iloc[idx,18]
        # shrm = self.df.iloc[idx,19]
        # eye_id = self.df.iloc[idx,22]
        # bcva = self.df.iloc[idx,23]
        # cst = self.df.iloc[idx,24]
        # patient = self.df.iloc[idx,25]
        return image, vit_deb,ir_hrf, full_vit,partial_vit,fluid_irf,drt,eye_id,bcva,cst,patient


class BiomarkerDatasetAll(data.Dataset):
    def __init__(self,df, img_dir, transforms):
        self.img_dir = img_dir
        self.transforms = transforms
        self.df = pd.read_csv(df)
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        current_name = self.df.iloc[idx, 0]
        path = self.img_dir + current_name
        # image = Image.open(path).convert("L")
        image = Image.open(path).convert("RGB")
        image = np.array(image)
        image = Image.fromarray(image)
        image = self.transforms(image)
        atrophy = self.df.iloc[idx,2]
        EZ = self.df.iloc[idx,3]
        DRIL = self.df.iloc[idx,4]
        IR_hemm = self.df.iloc[idx,5]
        ir_hrf = self.df.iloc[idx,6]
        partial_vit = self.df.iloc[idx,7]
        full_vit = self.df.iloc[idx,8]
        preret_tiss = self.df.iloc[idx,9]
        vit_deb = self.df.iloc[idx,10]
        vmt = self.df.iloc[idx,11]
        drt = self.df.iloc[idx,12]
        fluid_irf = self.df.iloc[idx,13]
        fluid_srf = self.df.iloc[idx,14]

        rpe = self.df.iloc[idx,15]
        ga = self.df.iloc[idx,16]
        shrm = self.df.iloc[idx,17]
        eye_id = self.df.iloc[idx,18]
        bcva = self.df.iloc[idx,19]
        cst = self.df.iloc[idx,20]
        patient = self.df.iloc[idx,21]

        all_info = {
            "image": image,
            "atrophy": atrophy,
            "EZ": EZ,
            "DRIL": DRIL,
            "IR_hemm": IR_hemm,
            "ir_hrf": ir_hrf,
            "partial_vit": partial_vit,
            "full_vit": full_vit,
            "preret_tiss": preret_tiss,
            "vit_deb": vit_deb,
            "vmt": vmt,
            "drt": drt,
            "fluid_irf": fluid_irf,
            "fluid_srf": fluid_srf,
            "rpe": rpe,
            "ga": ga,
            "shrm": shrm,
            "eye_id": eye_id,
            "bcva": bcva,
            "cst": cst,
            "patient": patient
        }
        return all_info

if __name__ == '__main__':
    df = "/home/gu721/yzc/data/ophthalmic_multimodal/OLIVES_Dataset_Labels/ml_centric_labels/Biomarker_Clinical_Data_Images.csv"
    img_dir = "/home/gu721/yzc/data/ophthalmic_multimodal/OLIVES"
    dataset = BiomarkerDatasetAttributes(df=df, img_dir=img_dir, transforms=None)
    print(dataset[0])