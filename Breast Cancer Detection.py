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

print("-------------Missing Values-------------")
print(dataset.isnull().values.any())
print(dataset.isnull().values.sum())
print(dataset.columns[dataset.isnull().any()])
print(len(dataset.columns[dataset.isnull().any()]))
print(dataset['Unnamed: 32'].count())
dataset = dataset.drop(columns='Unnamed: 32')
print(dataset.shape)

print("-------------Categorical Values-------------")
print(dataset.select_dtypes(include='object').columns)
print(dataset['diagnosis'].unique())
print(dataset['diagnosis'].nunique())

dataset = pd.get_dummies(data=dataset,drop_first=True)

dataset['diagnosis_M'] = dataset['diagnosis_M'].astype(int)
print(dataset['diagnosis_M'].dtype)
print(dataset['diagnosis_M'].unique())

print(dataset.head())


print("-------------Count Plot-------------")
sns.countplot(x=dataset['diagnosis_M'], label='Count')
plt.xticks([0, 1], ['0', '1'])
plt.show()

#print((dataset.diagnosis_M == 0).sum())