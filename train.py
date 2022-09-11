from glob import glob
import os
from utils import create_dir
import numpy as np
import tensorflow as tf
import constants as ct
import cv2
import pandas as pd
from model import ModelCreation
from keras.losses import CategoricalCrossentropy
from keras.optimizers import Adam
from keras.metrics import Precision, Recall, AUC, accuracy
from tensorflow.python.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard

class Training():
    
    
    
    def __init__(self) -> None:
        """ Seeding """
        np.random.seed(ct.RANDOM_STATE)
        tf.random.set_seed(ct.RANDOM_STATE)
        
        # """ Directory for string files """
        # create_dir(os.path.join(ct.SOURCE_PATH,ct.FILES))
        
        """ Hyperparameters """
        batch_size = 64
        lr = 1e-4
        num_epochs = 100
        self.model_path = os.path.join(ct.MODELS, ct.FILES, "model.h5")
        self.csv_path = os.path.join(ct.MODELS, ct.FILES, "data.csv")
    
    
        
        """ Dataset """
        # self.train_csv_path = os.path.join(ct.SOURCE_PATH + ct.NEW_DATA, "train")
        # self.valid_csv_path = os.path.join(ct.SOURCE_PATH + ct.NEW_DATA, "test")
        
        train_x_path, self.train_csv = self.load_data(ct.TRAIN)
        # train_x_path, train_y_path = shuffling(train_x_path, train_y_path)
        valid_x_path, self.val_csv = self.load_data(ct.VAL)
        
        print(f"Train: {len(train_x_path)}")
        print(f"Valid: {len(valid_x_path)}")
        
        train_dataset = self.tf_dataset(train_x_path, self.train_csv, batch=batch_size)
        valid_dataset = self.tf_dataset(valid_x_path, self.val_csv,  batch=batch_size)
        
        # for features, targets in train_dataset.take(1):
        #     print ('Features: {}, Target: {}'.format(features, targets))  
    
        """ Model """
        mc = ModelCreation()
        self.model = mc.custom_model((ct.H, ct.W, 3), 6)
        self.model.compile(loss=CategoricalCrossentropy(), optimizer = Adam(lr), metrics=[accuracy, AUC(), Recall(), Precision()])
        
        callbacks = [
            ModelCheckpoint(self.model_path, verbose = 1, save_best_only = True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=5, min_lr=1e-7, verbose=1),
            CSVLogger(self.csv_path),
            TensorBoard(),
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=False),
        ]
        
        self.model.fit(
            train_dataset,
            epochs=num_epochs,
            validation_data=valid_dataset,
            callbacks=callbacks
        )

        
    def read_image(self, path):
        path = path.decode()
        x = cv2.imread(path, cv2.B)
        x = x/255.0
        x = x.astype(np.float32)
        return x

    def load_data(self, train_val):
        x = sorted(glob(os.path.join(ct.DATASETS, train_val, ct.IMAGES, "*.jpg")))
        y = pd.read_csv(os.path.join(ct.DATASETS, train_val, ct.IMAGES+".csv"))
        self.check([x1.split("/")[-1].split(".")[0] for x1 in x], y.image.values.tolist())
        return x,y

    def check(self,a1, a2):
        if len(a1) != len(a2):
            print("Wrong Length")
        for i in range(len(a1)):
            if a1[i] != a2[i]:
                print("Wrong")

    def tf_parse(self, x_path, y):
        x = tf.numpy_function(self.read_image, [x_path], tf.float32)
        x.set_shape([ct.H, ct.W, 3])
        return x, y

    def tf_dataset(self, X, Y, batch = 2):
        dataset = tf.data.Dataset.from_tensor_slices((X, Y.iloc[:,2:8].values))
        dataset = dataset.map(self.tf_parse)
        dataset = dataset.batch(batch_size=batch)
        dataset = dataset.prefetch(10)
        return dataset


a = Training()