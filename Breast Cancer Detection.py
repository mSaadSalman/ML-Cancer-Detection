import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('/Users/saadsalman/Documents/GitHub/ML-Cancer-Detection/breast-cancer_data.csv')

#print(dataset.head())
#print(dataset.shape)
#print(dataset.info())
#print(dataset.select_dtypes(include='object').columns)

#print(dataset.describe())
#print(dataset.columns)

print(dataset.isnull().values.any())

print(dataset.isnull().values.sum())

print(dataset.columns[dataset.isnull().any()])

print(len(dataset.columns[dataset.isnull().any()]))

print(dataset['Unnamed: 32'].count())

dataset = dataset.drop(columns='Unnamed: 32')

print(dataset.shape)