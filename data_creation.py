from glob import glob
import os
import shutil
from tqdm import tqdm
import pandas as pd
import numpy as np
import constants as ct
import cv2
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from albumentations import HorizontalFlip, CoarseDropout, Blur, VerticalFlip, Rotate
from utils import create_dir
from sklearn.preprocessing import OneHotEncoder

class DataCreation:
    def __init__(self) -> None:
        """Seeding"""
        np.random.seed(ct.RANDOM_STATE)
        
        self.train_image_folder = os.path.join(ct.DATASETS, ct.TRAIN, ct.IMAGES)
        self.test_image_folder = os.path.join(ct.DATASETS, ct.VAL, ct.IMAGES)
        
        """ Create directories to save the augmented data """
        create_dir(self.train_image_folder)
        create_dir(self.test_image_folder)
        
    
            
    def load_data(self, train_folder, split_ratio=0.1):
        """Loading Images and CSV Data"""
        images = sorted(glob(os.path.join(train_folder, "*.jpg")))
        output = pd.read_csv("{}.csv".format(train_folder))
        output = self.pre_processing(output)
        self.oh = OneHotEncoder()
        self.oh.fit(output.name.values.reshape((-1, 1)))
        
        split_size = int((len(images)) * split_ratio)
        train_images, test_images = train_test_split(images,
                                        test_size=split_size,
                                        random_state=ct.RANDOM_STATE)
        train_csv = output[output["image"].isin([path.split("/")[-1].split(".")[0] for path in train_images])]
        test_csv = output[output["image"].isin([path.split("/")[-1].split(".")[0] for path in test_images])]
        return train_images, train_csv, test_images, test_csv
        
            
    """ Data Agumentation and Storing"""
    def augment_data(self, images, csv_data, train_val, augment=True):
        gen_data = pd.DataFrame(columns=csv_data.columns[0:2])
        for (x) in tqdm(images, total=len(images)):
            """Extract the name"""
            
            image_name = x.split("/")[-1].split(".")[0]
            
            """Reading the image"""
            x_img = cv2.imread(x, cv2.IMREAD_COLOR)
            
            y = str(csv_data[csv_data['image'] == image_name].name.values[0])
            X = [x_img]
            Y = [y]
            
            """Augmentation"""
            if augment:
                aug = HorizontalFlip(p=1)
                augmented = aug(image=x_img)
                X.append(augmented["image"])
                Y.append(y)
                
                aug = VerticalFlip(p=1)
                augmented = aug(image=x_img)
                X.append(augmented["image"])
                Y.append(y)
                
                aug = CoarseDropout(p=1, min_holes=3, max_holes=10, max_height=32, max_width=32)
                augmented = aug(image=x_img)
                X.append(augmented["image"])
                Y.append(y)
                
                aug = Blur(blur_limit=7, p=1.0)
                augmented = aug(image=x_img)
                X.append(augmented["image"])
                Y.append(y)
                
                aug = Rotate(limit=45, p=1.0)
                augmented = aug(image=x_img)
                X.append(augmented["image"])
                Y.append(y)
                
            index = 0
            for x, y in zip(X, Y):

                image_path = os.path.join(ct.DATASETS, train_val, ct.IMAGES, f"{image_name}_{index}.jpg")
                x = cv2.resize(x, (ct.W, ct.H))
                cv2.imwrite(image_path, x, )
                gen_data.loc[len(gen_data)] = ["{}_{}".format(image_name,index),y]
                
                index += 1
        
        s = self.oh.fit_transform(gen_data["name"].values.reshape(-1,1))
        gen_data = pd.concat([gen_data,pd.DataFrame(s.toarray())] ,axis=1)
        gen_data = gen_data.sort_values(by = ["image"])
        gen_data.to_csv(os.path.join(ct.DATASETS, train_val, "{}.csv".format(ct.IMAGES)), index=False)
        
    def pre_processing(self, df):
        df  = df[['image', 'name']]
        return df
        
        
dc = DataCreation()

train_images, train_csv, test_images, test_csv = dc.load_data("data/butterfly_mimics/images")
print("Training Image Size: {}, Testing Image Size: {}".format(len(train_images), len(test_images)))
print("Training CSV Shape: {}, Testing CSV Shape: {}".format(train_csv.shape, test_csv.shape))
dc.augment_data(train_images, train_csv, ct.TRAIN, False)
dc.augment_data(test_images, test_csv, ct.VAL, False)