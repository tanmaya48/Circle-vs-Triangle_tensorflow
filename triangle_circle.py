#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
##from tqdm import tqdm


#   creates image training database
#   all images with same labels in separate folders
#   also images in grayscale
#    
#

def create_training_data(DATADIR,CATEGORIES):
    
    training_data = []
    
    for category in CATEGORIES:  # 

        path = os.path.join(DATADIR,category)  # create path to samples labelwise
        class_num = CATEGORIES.index(category)  # get the classification

        for img in os.listdir(path):  # iterate over each image per dogs and cats
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                
                training_data.append([img_array, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass
    return training_data    




DATADIR = "shapes/"
CATEGORIES = ["circle", "triangle"]

training_data = create_training_data(DATADIR,CATEGORIES)

DATADIR = "shapes/test_shapes/"

testing_data = create_training_data(DATADIR,CATEGORIES)

print(len(training_data))
print(len(testing_data))


# In[4]:


import random

random.shuffle(training_data)
X = []
Y = []
for features,label in training_data:
    X.append(features)
    Y.append(label)
    
# shuffle data to make it sorted    
    
random.shuffle(testing_data)
Xtest = []
Ytest = []
for features,label in testing_data:
    Xtest.append(features)
    Ytest.append(label)    
    
    

    
plt.imshow(training_data[16][0], cmap='gray')  # graph it
print(training_data[16][1])
plt.show()  # display!    


# In[5]:


import tensorflow as tf


# In[ ]:





# In[17]:


X = tf.keras.utils.normalize(X, axis=1)
Xtest = tf.keras.utils.normalize(Xtest, axis=1)

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(50, activation=tf.nn.relu))
          
model.add(tf.keras.layers.Dense(9, activation=tf.nn.relu))
          
model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))    
          
          
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])    
          
    
X = np.array(X).reshape(-1, 20, 20, 1)

Y = np.array(Y)    
    
          
model.fit(X, Y, epochs=40)          


# In[18]:




Xtest = np.array(Xtest).reshape(-1, 20, 20, 1)

Ytest = np.array(Ytest)  

val_loss, val_acc = model.evaluate(Xtest, Ytest)
print(val_loss)
print(val_acc)


# In[6]:





# In[7]:





# In[ ]:




