# -*- coding: utf-8 -*-
"""Copy of mnist-cnn.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/18dnA0xD3HShQNqv6VlrjUxnJNUS2h2yx
"""



"""# Loading Datasets and Importing Libraries """

# Commented out IPython magic to ensure Python compatibility.
# Importing Tensorflow and the required visualization libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

#tf.config.run_functions_eagerly(True)
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns

#Loading the Dataset
train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')

# Seperating the independent feature as y
y = train['label']
train = train.drop('label', axis =1)

"""# Visualizing Data"""

#Visualizing the Distribution of digits in labels
#sns.countplot(y)

'''
#Visualing an example 
img = train.iloc[10].to_numpy()
img = img.reshape((28,28))
plt.imshow(img,cmap='gray')
plt.title(y.iloc[10])
plt.axis("off")
plt.show()
'''

"""# Preprocessing Data"""

#Preprocessing the Data
train=train/225.0
test = test/225.0
train = np.array(train)
test= np.array(test)
train = train.reshape(train.shape[0], 28, 28,1)
test = test.reshape(test.shape[0], 28, 28,1)

#Splitting the data into training and validation 
from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(train, y, test_size=0.2)

#Converting the train and validation labels to one-hot encodings
Y_train = tf.keras.utils.to_categorical(Y_train, num_classes=10)
Y_val = tf.keras.utils.to_categorical(Y_val, num_classes=10)

"""# Building, Compiling and Training model"""

#Preparing a CNN model architecture
model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (5,5), activation='relu', kernel_initializer='he_uniform',input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Dropout(0.2),
            
            tf.keras.layers.Conv2D(64, (5,5), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),    
            tf.keras.layers.Dropout(0.25),
            
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(10, activation='softmax')
    ])


#Getting the model framework/summary 
#from keras.utils.vis_utils import plot_model
#plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
#from IPython.display import Image
#Image("model.png")

#Compiling the model
model.compile(optimizer= tf.keras.optimizers.SGD(lr=0.1, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])


#Data augmentation to prevent overfitting
datagen = tf.keras.preprocessing.image.ImageDataGenerator()


#History Logging
#from keras.callbacks import CSVLogger
#csv_logger = CSVLogger("model_history_log.csv", append=True)

#Training the model
history = model.fit_generator(datagen.flow(X_train, Y_train,batch_size=64),validation_data=(X_val, Y_val)
    ,epochs=15,steps_per_epoch=X_train.shape[0] // 64)

"""# Evaluating Results"""


#Comparing losses and accuraries 
plt.plot(history.history['loss'], color='r', label="Train_Loss")
plt.plot(history.history['val_loss'], color='b', label="Val_Loss")
plt.legend(loc="upper right")
plt.show()
plt.plot(history.history['accuracy'], color='r', label="Train_Acc")
plt.plot(history.history['val_accuracy'], color='b', label="Val_Acc")
plt.legend(loc="lower right")
plt.show()

'''
#Plotting Confusion Matrix
y_pred1 = model.predict(X_val)
y_pred1 = np.argmax(y_pred1, axis=1)
y_true = np.argmax(Y_val, axis=1)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred1)
print(cm)
'''

"""# Saving the Predictions"""


#Predicting and Saving it as a CSV file
y_pred = model.predict(test)
y_pred = np.argmax(y_pred, axis=1)
y_pred = pd.Series(y_pred, name='Label')
sub = pd.concat([pd.Series(range(1, 28001), name="ImageId"), y_pred], axis=1)
sub.to_csv('./RESULT_1(e)_.csv', index=False)