import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
import constants as ct
from keras.models import load_model

def load_data(train_val):
    x = sorted(glob(os.path.join(ct.DATASETS, train_val, ct.IMAGES, "*.jpg")))
    y = pd.read_csv(os.path.join(ct.DATASETS, train_val, ct.IMAGES+".csv"))
    return x,y
        
# def save_results(image, mask, y_pred, save_image_path):
#     ## i - m - yp - yp*i
#     line = np.ones((ct.H, 10, 3)) * 128

#     mask = np.expand_dims(mask, axis=-1)    ## (512, 512, 1)
#     mask = np.concatenate([mask, mask, mask], axis=-1)  ## (512, 512, 3)
#     mask = mask * 255

#     y_pred = np.expand_dims(y_pred, axis=-1)    ## (512, 512, 1)
#     y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1)  ## (512, 512, 3)

#     masked_image = image * y_pred
#     y_pred = y_pred * 255

#     cat_images = np.concatenate([image, line, mask, line, y_pred, line, masked_image], axis=1)
#     cv2.imwrite(save_image_path, cat_images)

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(ct.RANDOM_STATE)
    tf.random.set_seed(ct.RANDOM_STATE)

    """ Directory for storing files """
    # create_dir(ct.SOURCE_PATH + ct.RESULTS)

    """ Loading model """
    # with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
    model = load_model(os.path.join(ct.MODELS , ct.FILES , "model.h5"))
        
    """ Load the dataset """
    # valid_path = os.path.join(ct.SOURCE_PATH + ct.NEW_DATA, "test")
    test_x, csv_y = load_data(ct.VAL)
    print(f"Test: {len(test_x)}")
    
    """ Evaluation and Prediction """
    SCORE = []
    for x in tqdm(test_x, total=len(test_x)):
        """ Extract the name """
        name = x.split("/")[-1].split(".")[0]
        y  = csv_y[csv_y["image"] == name].iloc[:,2:9]
        """ Reading the image """
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        x = image/255.0
        x = np.expand_dims(x, axis=0)
        
        """ Prediction """
        y = model.predict(x)[0]
        # y_pred = np.squeeze(y_pred, axis=-1)
        # y_pred = y_pred > 0.5
        # y_pred = y_pred.astype(np.int32)
        
        """ Saving the prediction """
        save_image_path = f"{ct.SOURCE_PATH + ct.RESULTS}/{name}.png"
        # save_results(image, mask, y_pred, save_image_path)
        
        """ Flatten the array """
        mask = mask.flatten()
        y_pred = y_pred.flatten()
        
        """ Calculating the metrics values """
        acc_value = accuracy_score(mask, y_pred)
        f1_value = f1_score(mask, y_pred, labels=[0, 1], average="binary")
        jac_value = jaccard_score(mask, y_pred, labels=[0, 1], average="binary")
        recall_value = recall_score(mask, y_pred, labels=[0, 1], average="binary")
        precision_value = precision_score(mask, y_pred, labels=[0, 1], average="binary")
        SCORE.append([name, acc_value, f1_value, jac_value, recall_value, precision_value])
    
    """ Metrics values """
    score = [s[1:]for s in SCORE]
    score = np.mean(score, axis=0)
    print(f"Accuracy: {score[0]:0.5f}")
    print(f"F1: {score[1]:0.5f}")
    print(f"Jaccard: {score[2]:0.5f}")
    print(f"Recall: {score[3]:0.5f}")
    print(f"Precision: {score[4]:0.5f}")
    
    df = pd.DataFrame(SCORE, columns=["Image", "Accuracy", "F1", "Jaccard", "Recall", "Precision"])
    df.to_csv(ct.SOURCE_PATH + ct.FILES +"/score.csv")
