import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt






#Comparing losses and accuraries 
plt.plot(history.history['loss'], color='r', label='Train Loss')
plt.plot(history.history['val_loss'], color='b', label='Val Loss')
plt.show()


'''
plt.plot(history.history['accuracy'], color='r', label='Train Acc')
plt.plot(history.history['val_accuracy'], color='b', label='Val Acc')
plt.show()
'''