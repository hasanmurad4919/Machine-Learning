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
X_train, X_test, y_train, y_test = train_test_split(dataset_feature, dataset_target, test_size = 0.2, random_state=0)
scaler = MinMaxScaler()
scaler.fit(X_train)


X_train_scaled = scaler.transform(X_train)

X_test_scaled = scaler.transform(X_test)

#applying logistic regression & demonstrating the result
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

model = LogisticRegression()
model.fit(X_train_scaled, y_train) #Training the model
predictions = model.predict(X_test_scaled)
#print(predictions)# printing predictions
logistic_score=accuracy_score(y_test, predictions)
print("logistic regression=",logistic_score)


#applying Decision Tree & demonstrating the result
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(criterion='entropy',random_state=1)
clf.fit(X_train_scaled, y_train)
y_pred = clf.predict(X_test_scaled)
Tree_score=accuracy_score(y_pred,y_test)
print("Decision Tree=",Tree_score)

#comparing scores by Barchart
import seaborn as sns
score_data={'logistic regression': [logistic_score], 'Decision Tree': [Tree_score]}
df = pd.DataFrame(score_data, columns = ['logistic regression', 'Decision Tree'])
sns.set_theme(style="whitegrid")
ax = sns.barplot(data=df)

