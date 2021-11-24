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

#applying Support vector classifier & demonstrating the result
print("Before Principle Component Analysis(PCA):")
print()
from sklearn.svm import SVC
svc = SVC(kernel="linear")
svc.fit(X_train_scaled, y_train)
print("Training accuracy of the Support vector classifier model is {:.2f}".format(svc.score(X_train_scaled, y_train)))
print("Testing accuracy of the Support vector classifier model is {:.2f}".format(svc.score(X_test_scaled, y_test)))
svc1=svc.score(X_test_scaled, y_test)
print()

#applying Random Forest classifier & demonstrating the result
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=50)
rfc.fit(X_train_scaled, y_train)
print("The Training accuracy of the Random Forest model is {:.2f}".format(rfc.score(X_train_scaled, y_train)))
print("The Testing accuracy of the Random Forest model is {:.2f}".format(rfc.score(X_test_scaled, y_test)))
rf1=rfc.score(X_test_scaled, y_test)
print()

#applying Neural Network Classifier classifier & demonstrating the result
from sklearn.neural_network import MLPClassifier
nnc=MLPClassifier(hidden_layer_sizes=(7), activation="relu", max_iter=10000)
nnc.fit(X_train_scaled, y_train)
print("The Training accuracy of the Neural Network model is {:.2f}".format(nnc.score(X_train_scaled, y_train)))
print("The Testing accuracy of the Neural Network model is {:.2f}".format(nnc.score(X_test_scaled, y_test)))
nn1=nnc.score(X_test_scaled, y_test)
print()


#applying Principle Component Analysis(PCA)
print("After Principle Component Analysis(PCA):")
print()
from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
from sklearn.decomposition import PCA
pca=PCA(n_components=6)
data=pca.fit_transform(scaler.fit_transform(dataset_feature))
main_df=pd.DataFrame(data=dataset_feature, columns=dataset_feature.columns)
main_df["target"]=dataset_target
main_df.head()
X= main_df.drop("target", axis=1)
y=main_df["target"]
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X, y , test_size=0.2 , random_state=42)

#applying Support vector classifier & demonstrating the result
svc = SVC(kernel="linear")
svc.fit(X_train_scaled, y_train)
print("Training accuracy of the Support vector classifier model is {:.2f}".format(svc.score(X_train_scaled, y_train)))
print("Testing accuracy of the Support vector classifier model is {:.2f}".format(svc.score(X_test_scaled, y_test)))
svc2=svc.score(X_test_scaled, y_test)
print()

#applying Random Forest classifier & demonstrating the result
rfc = RandomForestClassifier(n_estimators=50)
rfc.fit(X_train_scaled, y_train)
print("The Training accuracy of the Random Forest model is {:.2f}".format(rfc.score(X_train_scaled, y_train)))
print("The Testing accuracy of the Random Forest model is {:.2f}".format(rfc.score(X_test_scaled, y_test)))
rf2=rfc.score(X_test_scaled, y_test)
print()

#applying Neural Network Classifier classifier & demonstrating the result
nnc=MLPClassifier(hidden_layer_sizes=(7), activation="relu", max_iter=10000)
nnc.fit(X_train_scaled, y_train)
print("The Training accuracy of the Neural Network model is {:.2f}".format(nnc.score(X_train_scaled, y_train)))
print("The Testing accuracy of the Neural Network model is {:.2f}".format(nnc.score(X_test_scaled, y_test)))
nn2=nnc.score(X_test_scaled, y_test)
print()

#comparing scores by Barchart
import seaborn as sns
score_data={'SVC(Before PCA)': [svc1], 'SVC(After PCA)': [svc2],'Random Forest(Before PCA)': [rf1], 'Random Forest(After PCA)': [rf2],'Neural Network(Before PCA)': [nn1], 'Neural Network(After PCA)': [nn2]}
df = pd.DataFrame(score_data, columns = ['SVC(Before PCA)', 'SVC(After PCA)','Random Forest(Before PCA)','Random Forest(After PCA)','Neural Network(Before PCA)','Neural Network(After PCA)'])
sns.set_theme(style="whitegrid")
ax = sns.barplot(data=df)
plt.setp(ax.get_xticklabels(), rotation=90)
