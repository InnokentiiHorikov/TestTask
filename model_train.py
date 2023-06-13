imort numpy as np

from keras import Sequential, Model
from keras import layers
from keras import Input
import keras.backend as K
import cv2
import os

train_image_dir = '/kaggle/input/airbus-ship-detection/train_v2/'

def decode_ep(enc_pix):

        enc_pix = np.asarray(enc_pix.split(), dtype = np.uint32)
        
        num_of_pix = np.array(enc_pix[::2])
        rep_of_pix = np.array(enc_pix[1::2])
        end_point = len(num_of_pix)
        arr_of_pix = [np.asarray(
          np.arange(num_of_pix[i], num_of_pix[i] + rep_of_pix[i]))
                  for i in range(end_point)]
        
        size = 768
        rle = np.concatenate(arr_of_pix)
        rle -= 1
        square = np.power(size, 2)
        plot_mask = np.zeros(square, dtype = np.uint8)
        plot_mask[rle] = 255
        plot_mask = np.reshape(plot_mask, (size, size)).T
        plot_mask = plot_mask / 255
        return plot_mask 

def train_gererator(tup, batch_size):
    out_rgb = []
    out_mask = []
    while True:
        for c_img_id, c_masks in tup:
            rgb_path = os.path.join(train_image_dir, c_img_id)
            c_img = cv2.imread(rgb_path, cv2.IMREAD_GRAYSCALE)
            c_mask = np.expand_dims(decode_ep(c_masks), -1)
            out_rgb += [c_img]
            out_mask += [c_mask]
            if len(out_rgb)>=batch_size:
                yield np.stack(out_rgb, 0)/255.0, np.stack(out_mask, 0)
                out_rgb, out_mask=[], []

def build_model(input_layer, start_neurons):
    
    conv1 = layers.Conv2D(2, (3, 3), activation="relu", padding = 'same')(input_layer)
    conv1 = layers.Conv2D(2, (3, 3), activation="relu", padding = 'same')(conv1)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)
    pool1 = layers.Dropout(0.5)(pool1)

    conv2 = layers.Conv2D(2, (3, 3), activation="relu", padding = 'same')(pool1)
    conv2 = layers.Conv2D(2, (3, 3), activation="relu", padding = 'same')(conv2)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)
    pool2 = layers.Dropout(0.5)(pool2)

    conv3 = layers.Conv2D(2, (3, 3), activation="relu", padding = 'same')(pool2)
    conv3 = layers.Conv2D(2, (3, 3), activation="relu", padding = 'same')(conv3)
    pool3 = layers.MaxPooling2D((2, 2))(conv3)
    pool3 = layers.Dropout(0.5)(pool3)

    conv4 = layers.Conv2D(2, (3, 3), activation="relu", padding = 'same')(pool3)
    conv4 = layers.Conv2D(2, (3, 3), activation="relu", padding = 'same')(conv4)
    pool4 = layers.MaxPooling2D((2, 2))(conv4)
    pool4 = layers.Dropout(0.5)(pool4)

    convm = layers.Conv2D(2, (3, 3), activation="relu", padding = 'same')(pool4)
    convm = layers.Conv2D(2, (3, 3), activation="relu", padding = 'same')(convm)

    deconv4 = layers.Conv2DTranspose(2, (3, 3), strides=(2, 2), padding = 'same')(convm)
    uconv4 = layers.concatenate([deconv4, conv4])
    uconv4 = layers.Dropout(0.5)(uconv4)
    uconv4 = layers.Conv2D(2, (3, 3), activation="relu", padding = 'same')(uconv4)
    uconv4 = layers.Conv2D(2, (3, 3), activation="relu", padding = 'same')(uconv4)

    deconv3 = layers.Conv2DTranspose(2, (3, 3), strides=(2, 2), padding = 'same')(uconv4)
    uconv3 = layers.concatenate([deconv3, conv3])
    uconv3 = layers.Dropout(0.5)(uconv3)
    uconv3 = layers.Conv2D(2, (3, 3), activation="relu", padding = 'same')(uconv3)
    uconv3 = layers.Conv2D(2, (3, 3), activation="relu", padding = 'same')(uconv3)

    deconv2 = layers.Conv2DTranspose(2, (3, 3), strides=(2, 2), padding = 'same')(uconv3)
    uconv2 = layers.concatenate([deconv2, conv2])
    uconv2 = layers.Dropout(0.5)(uconv2)
    uconv2 = layers.Conv2D(2, (3, 3), activation="relu", padding = 'same')(uconv2)
    uconv2 = layers.Conv2D(2, (3, 3), activation="relu", padding = 'same')(uconv2)

    deconv1 = layers.Conv2DTranspose(2, (3, 3), strides=(2, 2), padding = 'same')(uconv2)
    uconv1 = layers.concatenate([deconv1, conv1])
    uconv1 = layers.Dropout(0.5)(uconv1)
    uconv1 = layers.Conv2D(2, (3, 3), activation="relu", padding = 'same')(uconv1)
    uconv1 = layers.Conv2D(2, (3, 3), activation="relu", padding = 'same')(uconv1)
    output_layer = layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(uconv1)
    
    return output_layer


def DICE_score(y_true, y_pred):
    y_truef=K.flatten(y_true)
    y_predf=K.flatten(y_pred)
    And=K.sum(y_truef* y_predf)
    return((2* And + 100) / (K.sum(y_truef) + K.sum(y_predf) + 100))

def DICE_score_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


if __name__  == '__main__':
    model_traint()
