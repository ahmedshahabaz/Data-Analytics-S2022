import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt


df_data = pd.read_csv("./baseball_data_2005.csv")
df_data.dropna()

df_train = df_data

df_train['N_H'] = df_train.iloc[:, -6:].sum(axis=1)

# as we have the N_H (# of Hits) calculated and stored in a separate column.
# dropping the unnecessary columns. We only need N (Season AB) and N_H

df_train.drop(df_train.columns[[2, 4, 5, 6, 7 , 8, 9, 10, 11, 12, 13, 14, 15]] , axis = 1, inplace = True)


N_H = df_train.iloc[:, 3].values
N = df_train.iloc[:, 2].values


# Maximum Likelihood Estimation (MLE)

MLE = N_H / N

# Parameters of Beta distribution as given in the question

a_0 = 100
b_0 = 300


# Please refer to the report for the Maximum a Posteriori (MAP) parameter estimation formula

MAP = (N_H + a_0 - 1) / (a_0 + b_0 + N - 2)


df_train['MLE'] = MLE

df_train['MAP'] = MAP


df_train.rename(columns={'Season AB': 'Season AB / N'}, inplace = True)

print(df_train.iloc[:])

df_train.to_csv('./results.csv')




# Some corner cases

# Clifford	Bartosh	1	1	1	0.25062656641604
# Adam	Bernero		1	1	1	0.25062656641604

# Bronson	Arroyo	1	0	0	0.24812030075188
# James	Baldwin		1	0	0	0.24812030075188



# Some average case (MLE and MAP similar)

# Clint	Barmes		350	101	0.288571428571429	0.267379679144385
# Michael Barrett	424	117	0.275943396226415	0.262773722627737
# Jason	Bartlett	224	54	0.241071428571429	0.245980707395498
# Jayson Werth		337	79	0.234421364985163	0.242176870748299


