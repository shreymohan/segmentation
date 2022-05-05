import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import keras
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Input, Conv2DTranspose, Concatenate, BatchNormalization, UpSampling2D
from keras.layers import  Dropout, Activation
from tensorflow.keras.optimizers import Adam, SGD
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras import backend as K
from keras.utils.vis_utils import plot_model
import tensorflow as tf
import random
from random import shuffle

test_dir='/home/shrey/data/val/images'
model_path='./model/unet.h5'

model = keras.models.load_model(model_path)

test_files=os.listdir(test_dir)
path = np.random.choice(test_files)
raw = Image.open(os.path.join(test_dir,path))
raw = np.array(raw.resize((256, 256)))/255.
raw = raw[:,:,0:3]

#predict the mask 
pred = model.predict(np.expand_dims(raw, 0))

#mask post-processing 
msk  = pred.squeeze()
msk = np.stack((msk,)*3, axis=-1)
msk[msk >= 0.5] = 1 
msk[msk < 0.5] = 0 

#show the mask and the segmented image 
combined = np.concatenate([raw, msk, raw* msk], axis = 1)
plt.axis('off')
plt.imsave('./test_res.jpeg',combined)
#mask = Image.fromarray(msk)
#segment=Image.fromarray(raw*msk)
#mask.save("./image_mask.jpeg")
#segment.save('./image_seg.jpeg')
#plt.show()
