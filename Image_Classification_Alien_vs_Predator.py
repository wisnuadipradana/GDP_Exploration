#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install tensorflow


# In[2]:


pip install keras


# In[14]:


pip install wget


# In[1]:


# Importing the libraries
import tensorflow as tf
import os
import time
import numpy as np
import matplotlib as plt
from tensorflow import keras
from tensorflow.keras.optimizers import RMSprop
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile


# In[2]:


tf.__version__


# In[3]:


# Locating our data
dataset_dir = "D:\\Koding\\Python\\Dicoding Machine Learning\\data\\"
training_dir = dataset_dir + "train\\"
testing_dir = dataset_dir + "validation\\"


# In[4]:


# Designing the layers
CNN_model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=1, activation='relu', input_shape=(512, 512, 3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=512, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])
opt = keras.optimizers.RMSprop(learning_rate=0.001)
CNN_model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['acc'])


# In[5]:


get_ipython().run_cell_magic('time', '', "converter = ImageDataGenerator(rescale = 1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, validation_split=0.2)\ndata_train = converter.flow_from_directory(training_dir, target_size=(512,512), batch_size=1, class_mode='binary', subset='training')\ndata_val = converter.flow_from_directory(training_dir, target_size=(512,512), batch_size=32, class_mode='binary', subset='validation')\n\ntest_datagen = ImageDataGenerator(rescale = 1./255)\ndata_test = test_datagen.flow_from_directory(testing_dir, target_size = (512, 512), batch_size=16, class_mode = 'binary')")


# In[6]:


trained_model = CNN_model.fit(data_train, epochs=5, verbose=True, steps_per_epoch=15, validation_data = data_val)


# In[10]:


print("Mean_Train_Accuracy: ",np.mean(trained_model.history['acc']))
print("Max_Train_Accuracy: ",np.max(trained_model.history['acc']))

print("Mean_Validation_Accuracy: ",np.mean(trained_model.history['val_acc']))
print("Max_Validation_Accuracy: ",np.max(trained_model.history['val_acc']))

import matplotlib.pyplot as plt

def plot_graphs(history, string):
    plt.subplots(figsize=(12,8), dpi=100)
    plt.plot(history.history[string],'r')
    plt.plot(history.history['val_' + string], 'b')
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()

plot_graphs(trained_model, "acc")
plot_graphs(trained_model, "loss")


# In[12]:


get_ipython().run_cell_magic('time', '', '\n# plot ALIEN validation result\n\nimage_path = "D:\\\\Koding\\\\Python\\\\Dicoding Machine Learning\\\\data\\\\validation\\\\alien\\\\"\nname_file = os.listdir(image_path)\n\ncount = 0\nplt.figure(figsize=(15,15))\nfor i in range(100):\n    img = image.load_img(image_path+name_file[i], target_size=(512, 512))\n    x = image.img_to_array(img)\n    x = np.expand_dims(x, axis=0)\n\n    images = np.vstack([x])\n    classes = CNN_model.predict(images, batch_size=10)\n    if classes[0] > 0.5:\n        class_res = "predator"\n    else:\n        class_res = "alien"\n        count += 1\n        \n    plt.subplot(10,10,i+1)\n    plt.imshow(img)\n    plt.title(class_res)\n    plt.xticks([])\n    plt.yticks([])\n    \nplt.tight_layout()\nprint("Accuracy: ",count,"%")')


# In[13]:


get_ipython().run_cell_magic('time', '', '\n# plot PREDATOR validation result\n\ncount = 0\nplt.figure(figsize=(15,15))\nfor i in range(100):\n    image_path = "D:\\\\Koding\\\\Python\\\\Dicoding Machine Learning\\\\data\\\\validation\\\\predator\\\\"+str(i)+".jpg"\n    img = image.load_img(image_path, target_size=(512, 512))\n    x = image.img_to_array(img)\n    x = np.expand_dims(x, axis=0)\n\n    images = np.vstack([x])\n    classes = CNN_model.predict(images, batch_size=10)\n    if classes[0] > 0.5:\n        class_res = "predator"\n        count += 1\n    else:\n        class_res = "alien"   \n        \n    plt.subplot(10,10,i+1)\n    plt.imshow(img)\n    plt.title(class_res)\n    plt.xticks([])\n    plt.yticks([])\n    \nplt.tight_layout()\nprint("Accuracy: ",count,"%")')

