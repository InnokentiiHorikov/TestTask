imort numpy as np

from keras import Sequential, Model
from keras import layers
from keras import Input
from keras.preprocessing.image import ImageDataGenerator as IDG
import keras.backend as K
import cv2
import os


train_image_dir = '/kaggle/input/airbus-ship-detection/train_v2/'
train_masks_dir = '/kaggle/input/airbus-ship-detection/train_ship_segmentations_v2.csv'
#Parameters for ImageDataGenerator 
dg_args = dict(featurewise_center = False, 
                  samplewise_center = False,
                  rotation_range = 45, 
                  width_shift_range = 0.1, 
                  height_shift_range = 0.1, 
                  shear_range = 0.01,
                  zoom_range = [0.9, 1.25],  
                  horizontal_flip = True, 
                  vertical_flip = True,
                  fill_mode = 'reflect',
                   data_format = 'channels_last')


def read_data(train_masks_dir):
        #Returns a numpy array of encoded pixels group to corresponding image

        #Reading the masks
        train_data = pd.read_csv('/kaggle/input/airbus-ship-detection/train_ship_segmentations_v2.csv')
        #Left only images containing the ships
        train_data = train_data.dropna()
        #Grouping, cause some images contain few ships 
        train_data = train_data.groupby("ImageId")["EncodedPixels"].agg(EncodedPixels = lambda x: " ".join(map(str,  x)))
        train_data.reset_index("ImageId", inplace = True)
        train_data = train_data.to_numpy()
        return train_data
def decode_ep(enc_pix):
#        "Enc_pix - encoded pixels: string"
#        "Splitting string into separate integers"
        enc_pix = np.asarray(enc_pix.split(), dtype = np.uint32)
        
#        "Each odd number is start pixel, even number - a difference between "
#        "start and end pixel"
        num_of_pix = np.array(enc_pix[::2])
        rep_of_pix = np.array(enc_pix[1::2])
#       "Len of all arrray"
        end = len(num_of_pix)
        arr_of_pix = [np.asarray(
          np.arange(num_of_pix[i], num_of_pix[i] + rep_of_pix[i]))
                  for i in range(end)]
#        "resolution of image is 768 by 768"
        size = 768
#        "concatenating the separable arrays of decoded pixels"
        rle = np.concatenate(arr_of_pix)
        rle -= 1
        square = np.power(size, 2)
        
        plot_mask = np.zeros(square, dtype = np.uint8)
#        "As encoded pixels marks a ship, we can make a high contrast"
#        "between ship and sea or land(i.e no-ship zone)"
        plot_mask[rle] = 255
        plot_mask = np.reshape(plot_mask, (size, size)).T
        plot_mask = plot_mask / 255
        return plot_mask 

def train_gererator(tup, batch_size = None):
    """
    tup: tuple or list containing names with corresponding encoded pixels
    """
#    "We have to use generator due to large dataset"
    out_rgb = []
    out_mask = []
    while True:
        for c_img_id, c_masks in tup:
            rgb_path = os.path.join(train_image_dir, c_img_id)
            c_img = cv2.imread(rgb_path)
            c_mask = np.expand_dims(decode_ep(c_masks), -1)
#            "Reduction of resolution from 768x768 to 256x256"
            c_img = c_img[::3, ::3]
            c_mask = c_mask[::3, ::3]
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


def DICE_score(y_true, y_pred, eps = 0.1):
    """
    y_true - real mask - numpy matrix 
    y_pred - predicted mask - Tensor
    eps - some small number to make smoothness of predict and avoiding case 0/0
    """
    y_true,y_pred  =K.flatten(y_true), K.flatten(y_pred)
    
    intersection=K.sum(y_true* y_pred)
    union  = K.sum(y_true*y_true) + K.sum(y_pred*y_pred)
    
    return(2* intersection + eps) / (union + eps)


def DICE_score_loss(y_true, y_pred):
    return 1-DICE_score(y_true, y_pred)


def create_gen(dg_args, in_gen):
    """
            in_gen - tuple containing stacked images and corresponding
            stacked masks
    """
        label_gen = IDG(**dg_args)

        image_gen = IDG(**dg_args)

    for in_x, in_y in in_gen:
        seed = np.random.choice(range(9999))
        
        
        
        g_x = image_gen.flow(255*in_x, 
                             batch_size = in_x.shape[0], 
                             seed = seed, 
                             shuffle=True)
        g_y = image_gen.flow(in_y, 
                             batch_size = in_x.shape[0], 
                             seed = seed, 
                             shuffle=True)

        yield next(g_x)/255.0, next(g_y)

if __name__  == '__main__':
    model_train()
