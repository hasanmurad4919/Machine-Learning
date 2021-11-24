#*****  "RUN THIS CODE IN GOOGLE COLAB  CAUSE SPYDER DOES NOT IMPORT SKLEARN"  *****

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv('heart disease classification dataset.csv')


#dropping the rows where the chol values are null 
dataset = dataset.dropna(axis = 0, subset = ['chol'])

#imputing trestbps & thalach column's null values
from sklearn.impute import SimpleImputer
impute = SimpleImputer(missing_values=np.nan, strategy='mean')
impute.fit(dataset[['trestbps']])
dataset['trestbps'] = impute.transform(dataset[['trestbps']])
impute.fit(dataset[['thalach']])
dataset['thalach'] = impute.transform(dataset[['thalach']])

#binary encoding the target column
dataset['target'] = dataset['target'].map({'no':0,'yes':1}) 

#one hot encoding sex column
dataset = pd.get_dummies(dataset, columns=['sex'])
dataset = dataset.rename(columns={'sex_male': 'male'})
dataset = dataset.rename(columns={'sex_female': 'female'})
clist = ['Unnamed: 0','age','male','female','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','target']
dataset = dataset[clist] 


#Intuition & Correlation
import seaborn as sns
dataset_corr = dataset.corr()
dataset_corr
#sns.heatmap(dataset_corr, cmap = 'YlGnBu')
dataset=dataset.drop(['ca','exang','fbs'], axis = 1)




#Scaling data using minmax scaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

dataset_feature=dataset.drop(['target'], axis = 1)
dataset_target=dataset['target']
X_train, X_test, y_train, y_test = train_test_split(dataset_feature, dataset_target, test_size = 0.25, random_state=0)
scaler = MinMaxScaler()
scaler.fit(X_train)


X_train_scaled = scaler.transform(X_train)

X_test_scaled = scaler.transform(X_test)

#applying KNN algorithm & demonstrating the result

knn=KNeighborsClassifier()
knn.fit(X_train_scaled, y_train)

print("Scaled test set accuracy: {:.2f}".format(
    knn.score(X_test_scaled, y_test)))
