import numpy as np 
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import cv2

test_image_dir = '/kaggle/input/airbus-ship-detection/test_v2'
#Compiling the model
def model_compile(model, train_data, valid_x, valid_y)
  model.compile(optimizer=Adam(1e-3, decay=1e-6), loss=DICE_score_loss, metrics=["binary_accuracy", DICE_score])


#End fitting it by train data
  gen = create_gen(train_gererator(train_data, 16))
  model.fit_generator(gen, steps_per_epoch=20,
                    epochs = 8, validation_data=(valid_x, valid_y))

  return model

def prediction(model, test_image_dir):
    fig, m_axs = plt.subplots(10, 2)
    for (ax1, ax2), c_img_name in zip(m_axs, test_paths):
      c_path = os.path.join(test_image_dir, c_img_name)
      c_img = cv2.imread(c_path)
      c_img = cv2.resize(c_img,(256,256))
      first_img = np.expand_dims(c_img, 0)/255.0
      first_seg = model.predict(first_img)
      first_img[0][:,:,0] = (first_img[0][:,:,0]*0.7 + 0.5*first_seg[0, :, :, 0])
      result = np.array(np.clip(first_img[0]*255.,0,255),dtype=np.int32)
      ax1.imshow(result)
      ax1.set_title('Image')
      ax2.imshow(first_seg[0, :, :, 0], vmin = 0, vmax = 1)
      ax2.set_title('Prediction')

if __name__ == '__main__':
  model_inference()
