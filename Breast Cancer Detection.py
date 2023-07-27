import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,precision_score,recall_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier


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
#plt.show()

print("-------Correlation matrix and heatmap-------")
dataset_2 = dataset.drop(columns='diagnosis_M')
print(dataset_2.head())

print(dataset_2.corrwith(dataset['diagnosis_M']).plot.bar(figsize=(20,10,),title = 'Correlated with diagnosis_M', rot=45, grid=True))
#plt.show()

corr = dataset.corr() #represents the direct correlation value of each indepenent variable to itself
print(corr)

plt.figure(figsize=(20,10))
sns.heatmap(corr,annot=True)
#plt.show()

print("-------Splitting Data into train and test sets -------")
x = dataset.iloc[:,1:-1].values
y= dataset.iloc[:,-1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
print(x_train.shape)

print("-------Feature Scaling -------") #Following steps used to have all features on same scale
sc = StandardScaler() 
x_train= sc.fit_transform(x_train)
x_test = sc.transform(x_test)

print(x_train)
print(x_test)

print("-------Logistic Regression -------") 
classifier_lr = LogisticRegression(random_state=0)
print(classifier_lr.fit(x_train,y_train))

y_pred= classifier_lr.predict(x_test)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)

results = pd.DataFrame([['Logistic Regression', acc, f1, prec, rec]],
                       columns = ['Model', 'Accuracy', 'F1 Score', 'Precision', 'Recall'])
print(results)

print("-------Cross Validation -------") 
cm = confusion_matrix(y_test, y_pred)
print(cm)

accuracies = cross_val_score(estimator=classifier_lr, X=x_train, y=y_train,cv=10)
print("Accuracy is {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation is {:.2f} %".format(accuracies.std()*100))

print("-------Random forest Classifier-------")
classifier_rm = RandomForestClassifier(random_state=0)
print(classifier_rm.fit(x_train, y_train))

y_pred = classifier_rm.predict(x_test)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)

model_results = pd.DataFrame([['Random Forest', acc, f1, prec, rec]],
                       columns = ['Model', 'Accuracy', 'F1 Score', 'Precision', 'Recall'])

results = pd.concat([results, model_results], ignore_index=True)
print(results)

accuracies = cross_val_score(estimator=classifier_rm, X=x_train, y=y_train,cv=10)
print("Accuracy is {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation is {:.2f} %".format(accuracies.std()*100))
cm = confusion_matrix(y_test, y_pred)
print(cm)