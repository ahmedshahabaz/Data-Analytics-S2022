# importing libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


file_path = "./airline-passengers.csv"

df_data = pd.read_csv(file_path)

train_data = df_data.values

'''
fig, ax = plt.subplots()

df_data['Passengers'].plot(kind = 'bar', color ='blue')
df_data['Passengers'].plot(kind = 'line', marker='.', color ='red', ms = 10)
'''


x = range(0,train_data.shape[0])
y = train_data[:,1]


plt.plot(x, y, color='g')

# naming the x axis
plt.xlabel('Month')
# naming the y axis
plt.ylabel("# of Passengers in Thousand")
 
# giving a title to my graph
#plt.title('Miss Classification Rate plot')


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(x,y)


# function to show the plot
plt.show()







