imort numpy as np

from keras import Sequential, Model
from keras import layers
from keras import Input

def to_mask(rle):
  size = 768
  rle -= 1
  square = np.power(size, 2)
  plot = np.zeros(square, dtype = np.uint8)
  plot[rle] = 255
  plot = np.reshape(plot, (width, height)).T
  return plot

def decode_ep(img_name, enc_pix):
  if enc_pix == '0':
    return img_name, np.zeroes((768,768))
  
  else:
    enc_pix = enc_pix.split()
    end_point = np.len(enc_pix)
    num_of_pix = enc_pix[::2]
    rep_of_pix = enc_pix[1::2]
    arr_of_pix = [np.arange(
      np.arange(num_of_pix[i], num_of_pix[i] + rep_of_pix[i]))
                  for i in range(lenght)]
               

    return img_name, to_mask(np.concatenate(arr_of_pix))
  
    
def train_valid(data, split_point):
    import os, shutil
    
    
    img_names_train = [data[i] for i in range(split_point)]
    os.mkdir("/kaggle/working/train")
    for name in img_names_train:
        src = os.path.join("/kaggle/input/airbus-ship-detection/train_v2", name)
        dst = os.path.join("/kaggle/working/train/", name)
        shutil.copyfile(src, dst)
    
    stop_point = len(data)
    
    img_names_valid = [tup[i] for i in range(split_point,stop_point)]
    os.mkdir("/kaggle/working/valid")
    for name in img_names_valid:
        src = os.path.join("/kaggle/input/airbus-ship-detection/train_v2", name)
        dst = os.path.join("/kaggle/working/valid/", name)
        shutil.copyfile(src, dst)    
        
def keras_process():
    from keras.preprocessing.image import ImageDataGenerator as IDG
    train_datagen = IDG(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
    "/kaggle/working/",
    classes = ['train'],
    target_size=(768, 768),
    batch_size=32,
    class_mode='binary',color_mode = 'grayscale')


valid_generator = train_datagen.flow_from_directory(
    "/kaggle/working/",
    classes = ['valid'],
    target_size=(768, 768),
    batch_size=32,
    class_mode='binary',color_mode = 'grayscale')

    return train_generator, valid_generator

def build_model(input_layer, start_neurons):
    
    conv1 = layers.Conv2D(2, (3, 3), activation="relu")(input_layer)
    conv1 = layers.Conv2D(2, (3, 3), activation="relu")(conv1)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)
    pool1 = layers.Dropout(0.5)(pool1)

    conv2 = layers.Conv2D(2, (3, 3), activation="relu")(pool1)
    conv2 = layers.Conv2D(2, (3, 3), activation="relu")(conv2)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)
    pool2 = layers.Dropout(0.5)(pool2)

    conv3 = layers.Conv2D(2, (3, 3), activation="relu")(pool2)
    conv3 = layers.Conv2D(2, (3, 3), activation="relu")(conv3)
    pool3 = layers.MaxPooling2D((2, 2))(conv3)
    pool3 = layers.Dropout(0.5)(pool3)

    conv4 = layers.Conv2D(2, (3, 3), activation="relu")(pool3)
    conv4 = layers.Conv2D(2, (3, 3), activation="relu")(conv4)
    pool4 = layers.MaxPooling2D((2, 2))(conv4)
    pool4 = layers.Dropout(0.5)(pool4)

    convm = layers.Conv2D(2, (3, 3), activation="relu")(pool4)
    convm = layers.Conv2D(2, (3, 3), activation="relu")(convm)

    deconv4 = layers.Conv2DTranspose(2, (3, 3), strides=(2, 2))(convm)
    conv4crop = layers.Cropping2D(cropping=((3, 4), (3, 4)))(conv4)
    uconv4 = layers.concatenate([deconv4, conv4crop])
    uconv4 = layers.Dropout(0.5)(uconv4)
    uconv4 = layers.Conv2D(2, (3, 3), activation="relu")(uconv4)
    uconv4 = layers.Conv2D(2, (3, 3), activation="relu")(uconv4)

    deconv3 = layers.Conv2DTranspose(2, (3, 3), strides=(2, 2))(uconv4)
    conv3crop = layers.Cropping2D(cropping=((15, 15), (15, 15)))(conv3)
    uconv3 = layers.concatenate([deconv3, conv3crop])
    uconv3 = layers.Dropout(0.5)(uconv3)
    uconv3 = layers.Conv2D(2, (3, 3), activation="relu")(uconv3)
    uconv3 = layers.Conv2D(2, (3, 3), activation="relu")(uconv3)

    deconv2 = layers.Conv2DTranspose(2, (3, 3), strides=(2, 2))(uconv3)
    conv2crop = layers.Cropping2D(cropping=((37, 38), (37, 38)))(conv2)
    uconv2 = layers.concatenate([deconv2, conv2crop])
    uconv2 = layers.Dropout(0.5)(uconv2)
    uconv2 = layers.Conv2D(2, (3, 3), activation="relu")(uconv2)
    uconv2 = layers.Conv2D(2, (3, 3), activation="relu")(uconv2)

    deconv1 = layers.Conv2DTranspose(2, (3, 3), strides=(2, 2))(uconv2)
    conv1crop = layers.Cropping2D(cropping=((82, 83), (82, 83)))(conv1)
    uconv1 = layers.concatenate([deconv1, conv1crop])
    uconv1 = layers.Dropout(0.5)(uconv1)
    uconv1 = layers.Conv2D(2, (3, 3), activation="relu")(uconv1)
    uconv1 = layers.Conv2D(2, (3, 3), activation="relu")(uconv1)
    output_layer = layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(uconv1)
    
    return output_layer


def DICE_score(pred_y, test_y, eps = 0.1):
    pred_y, test_y = pred_y.flatten(), test_y.flatten()
    intersection =  np.sum(np.multiply(pred_y, test_y))
    union = np.sum(np.power(pred_y)) + np.sum(np.power(test_y)) + eps
    dice = 2 * intersection / union 
    
    return dice


if __name__  == '__main__':
    model_traint()
