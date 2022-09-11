import os
import tensorflow as tf
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Input, Flatten, Dense
from keras.models import Model
from keras.applications import ResNet50
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
class ModelCreation():
    def __init__(self) -> None:
        self.model = None
    
    def custom_model(self, shape, num_classes):
        inputs = Input(shape)
        
        encoder = ResNet50(weights="imagenet", include_top=False, input_tensor=inputs)
        
        for layer in encoder.layers:
            layer.trainable = False
        
        image_features = encoder.get_layer("conv4_block6_out").output
        
        x = Conv2D(filters=32, kernel_size=3, padding='same', use_bias=False)(image_features)
        x = Activation('relu')(x)
        
        
        x = Conv2D(filters=64, kernel_size=3, use_bias=False)(x)
        x = Activation('relu')(x)
        
        
        x = Conv2D(filters=128, kernel_size=3, use_bias=False)(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), strides=2)(x),
        
        x = Flatten()(x[0]),
        x = Dense(64, activation='relu')(x[0]),
        
        x = Dense(num_classes, activation="softmax")(x[0])
        
        self.model  = Model(inputs, x)
        
        return self.model

if __name__ == "__main__":
    mc = ModelCreation()
    model = mc.custom_model((224, 224, 3), 5)
    model.summary()