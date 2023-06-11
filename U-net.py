imort numpy as np

from keras import Sequential, Model
from keras import layers
from keras import Input


def U-net(input_layer, start_neurons):
    # 128 -> 64
    conv1 = layers.Conv2D(2, (3, 3), activation="relu")(input_layer)
    conv1 = layers.Conv2D(2, (3, 3), activation="relu")(conv1)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)


    # 64 -> 32
    conv2 = layers.Conv2D(2, (3, 3), activation="relu")(pool1)
    conv2 = layers.Conv2D(2, (3, 3), activation="relu")(conv2)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)


    # 32 -> 16
    conv3 = layers.Conv2D(2, (3, 3), activation="relu")(pool2)
    conv3 = layers.Conv2D(2, (3, 3), activation="relu")(conv3)
    pool3 = layers.MaxPooling2D((2, 2))(conv3)


    # 16 -> 8
    conv4 = layers.Conv2D(2, (3, 3), activation="relu")(pool3)
    conv4 = layers.Conv2D(2, (3, 3), activation="relu")(conv4)
    pool4 = layers.MaxPooling2D((2, 2))(conv4)


    # Middle
    convm = layers.Conv2D(2, (3, 3), activation="relu")(pool4)
    convm = layers.Conv2D(2, (3, 3), activation="relu")(convm)

    # 8 -> 16
    deconv4 = layers.Conv2DTranspose(2, (3, 3), strides=(2, 2))(convm)
    conv4crop = layers.Cropping2D(cropping=((3, 4), (3, 4)))(conv4)
    uconv4 = layers.concatenate([deconv4, conv4crop])

    uconv4 = layers.Conv2D(2, (3, 3), activation="relu")(uconv4)
    uconv4 = layers.Conv2D(2, (3, 3), activation="relu")(uconv4)

    # 16 -> 32
    deconv3 = layers.Conv2DTranspose(2, (3, 3), strides=(2, 2))(uconv4)
    conv3crop = layers.Cropping2D(cropping=((15, 15), (15, 15)))(conv3)
    uconv3 = layers.concatenate([deconv3, conv3crop])

    uconv3 = layers.Conv2D(2, (3, 3), activation="relu")(uconv3)
    uconv3 = layers.Conv2D(2, (3, 3), activation="relu")(uconv3)

    # 32 -> 64
    deconv2 = layers.Conv2DTranspose(2, (3, 3), strides=(2, 2))(uconv3)
    conv2crop = layers.Cropping2D(cropping=((37, 38), (37, 38)))(conv2)
    uconv2 = layers.concatenate([deconv2, conv2crop])

    uconv2 = layers.Conv2D(2, (3, 3), activation="relu")(uconv2)
    uconv2 = layers.Conv2D(2, (3, 3), activation="relu")(uconv2)

    # 64 -> 128
    deconv1 = layers.Conv2DTranspose(2, (3, 3), strides=(2, 2))(uconv2)
    conv1crop = layers.Cropping2D(cropping=((82, 83), (82, 83)))(conv1)
    uconv1 = layers.concatenate([deconv1, conv1crop])

    uconv1 = layers.Conv2D(2, (3, 3), activation="relu")(uconv1)
    uconv1 = layers.Conv2D(2, (3, 3), activation="relu")(uconv1)
    print(uconv1.output_shape())
    output_layer = layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(uconv1)
    
    return output_layer

def DICE_score(pred_y, test_y, eps = 0.1):
    pred_y, test_y = pred_y.flatten(), test_y.flatten()
    intersection =  np.sum(np.multiply(pred_y, test_y))
    union = np.sum(np.power(pred_y)) + np.sum(np.power(test_y)) + eps
    dice = 2 * intersection / union 
    
    return dice


if __name__  == '__main__':
    U-net()
