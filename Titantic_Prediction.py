# -*- coding: utf-8 -*-
# Auto-converted from Jupyter Notebook (.ipynb)
# Source: Titantic Prediction.ipynb

# %% [markdown] (cell 1)
# # Titanic - Machine Learning from Disaster

# %% [markdown] (cell 2)
# ### Predicting survival on the Titanic

# %% [markdown] (cell 3)
# ![](https://storage.googleapis.com/kaggle-competitions/kaggle/3136/logos/header.png)

# %% [markdown] (cell 4)
# #### Data Dictionary
# |Variable|Definition|Key|
# |--------|----------|---|
# |survival|	Survival|	0 = No, 1 = Yes|
# |pclass|	Ticket class|	1 = 1st, 2 = 2nd, 3 = 3rd|
# |sex|	Sex	|
# |Age|	Age in years|	
# |sibsp	|# of siblings / spouses aboard the Titanic	|
# |parch	|# of parents / children aboard the Titanic	|
# |ticket	|Ticket number	|
# |fare	|Passenger fare	|
# |cabin	|Cabin number	|
# |embarked	|Port of Embarkation|	C = Cherbourg, Q = Queenstown, S = Southampton|

# %% (cell 5)
#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %% (cell 6)
df = pd.read_csv('titanic_train.csv')
df.head()

# %% (cell 7)
df.shape

# %% [markdown] (cell 8)
# ## Data Preprocessing

# %% (cell 9)
#removing the columns
df = df.drop(columns=['PassengerId','Name','Cabin','Ticket'], axis= 1)

# %% (cell 10)
df.describe()

# %% (cell 11)
#checking data types
df.dtypes

# %% (cell 12)
#checking for unique value count
df.nunique()

# %% (cell 13)
#checking for missing value count
df.isnull().sum()

# %% [markdown] (cell 14)
# #### Refining the data

# %% (cell 15)
# replacing the missing values
df['Age'] =  df['Age'].replace(np.nan,df['Age'].median(axis=0))
df['Embarked'] = df['Embarked'].replace(np.nan, 'S')

# %% (cell 16)
#type casting Age to integer
df['Age'] = df['Age'].astype(int)

# %% (cell 17)
#replacing with 1 and female with 0
df['Sex'] = df['Sex'].apply(lambda x : 1 if x == 'male' else 0)

# %% [markdown] (cell 18)
# #### Categorising in groups i.e. Infant(0-5), Teen (6-20), 20s(21-30), 30s(31-40), 40s(41-50), 50s(51-60), Elder(61-100)

# %% (cell 19)
# creating age groups - young (0-18), adult(18-30), middle aged(30-50), old (50-100)
df['Age'] = pd.cut(x=df['Age'], bins=[0, 5, 20, 30, 40, 50, 60, 100], labels = ['Infant', 'Teen', '20s', '30s', '40s', '50s', 'Elder'])

# %% [markdown] (cell 20)
# ## Exploratory Data Analysis

# %% [markdown] (cell 21)
# #### Plotting the Countplot to visualize the numbers

# %% (cell 22)
# visulizing the count of the features
fig, ax = plt.subplots(2,4,figsize=(20,20))
sns.countplot(x = 'Survived', data = df, ax= ax[0,0])
sns.countplot(x = 'Pclass', data = df, ax=ax[0,1])
sns.countplot(x = 'Sex', data = df, ax=ax[0,2])
sns.countplot(x = 'Age', data = df, ax=ax[0,3])
sns.countplot(x = 'Embarked', data = df, ax=ax[1,0])
sns.histplot(x = 'Fare', data= df, bins=10, ax=ax[1,1])
sns.countplot(x = 'SibSp', data = df, ax=ax[1,2])
sns.countplot(x = 'Parch', data = df, ax=ax[1,3])

# %% [markdown] (cell 23)
# #### Visualizing the replationship between the features

# %% (cell 24)
fig, ax = plt.subplots(2,4,figsize=(20,20))
sns.countplot(x = 'Sex', data = df, hue = 'Survived', ax= ax[0,0])
sns.countplot(x = 'Age', data = df, hue = 'Survived', ax=ax[0,1])
sns.boxplot(x = 'Sex',y='Fare', data = df, hue = 'Pclass', ax=ax[0,2])
sns.countplot(x = 'SibSp', data = df, hue = 'Survived', ax=ax[0,3])
sns.countplot(x = 'Parch', data = df, hue = 'Survived', ax=ax[1,0])
sns.scatterplot(x = 'SibSp', y = 'Parch', data = df,hue = 'Survived', ax=ax[1,1])
sns.boxplot(x = 'Embarked', y ='Fare', data = df, ax=ax[1,2])
sns.pointplot(x = 'Pclass', y = 'Survived', data = df, ax=ax[1,3])

# %% [markdown] (cell 25)
# ## Data Preprocessing 2

# %% (cell 26)
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(['S','C','Q'])
df['Embarked'] = le.transform(df['Embarked'])

# %% (cell 27)
age_mapping = {
    'infant': 0,
    'teen': 1,
    '20s': 2,
    '30s': 3,
    '40s': 4,
    '50s': 5,
    'elder': 6}
df['Age'] = df['Age'].map(age_mapping)
df.dropna(subset=['Age'], axis= 0, inplace = True)

# %% [markdown] (cell 28)
# #### Coorelation Heatmap

# %% (cell 29)
sns.heatmap(df.corr(), annot= True)

# %% [markdown] (cell 30)
# #### Separating the target and independent variable

# %% (cell 31)
y = df['Survived']
x = df.drop(columns=['Survived'])

# %% [markdown] (cell 32)
# ## Model Training

# %% [markdown] (cell 33)
# ### Logistic Regression

# %% (cell 34)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr

# %% (cell 35)
lr.fit(x,y)
lr.score(x,y)

# %% [markdown] (cell 36)
# ### Decision Tree Classifier

# %% (cell 37)
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree

# %% (cell 38)
dtree.fit(x,y)
dtree.score(x,y)

# %% [markdown] (cell 39)
# ### Support Vector Machine (SVM)

# %% (cell 40)
from sklearn.svm import SVC
svm = SVC()
svm

# %% (cell 41)
svm.fit(x,y)
svm.score(x,y)

# %% [markdown] (cell 42)
# ### K-Nearest Neighbor

# %% (cell 43)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn

# %% (cell 44)
knn.fit(x,y)
knn.score(x,y)

# %% [markdown] (cell 45)
# #### From the above four model Decision Tree Classifier has the highest Training accuracy, so only Decision Tree Classifier will work on the Test Set.

# %% [markdown] (cell 46)
# ### Importing the test set

# %% (cell 47)
df2 = pd.read_csv('titanic_test.csv')
df2.head()

# %% (cell 48)
#removing the columns
df2 = df2.drop(columns=['PassengerId','Name','Cabin','Ticket'], axis= 1)

# %% [markdown] (cell 49)
# ## Data Preprocessing the Test set

# %% (cell 50)
df2['Age'] =  df2['Age'].replace(np.nan,df2['Age'].median(axis=0))
df2['Embarked'] = df2['Embarked'].replace(np.nan, 'S')

# %% (cell 51)
#type casting Age to integer
df2['Age'] = df2['Age'].astype(int)

# %% (cell 52)
#replacing with 1 and female with 0
df2['Sex'] = df2['Sex'].apply(lambda x : 1 if x == 'male' else 0)

# %% (cell 53)
df2['Age'] = pd.cut(x=df2['Age'], bins=[0, 5, 20, 30, 40, 50, 60, 100], labels = [0,1,2,3,4,5,6])

# %% (cell 54)
le.fit(['S','C','Q'])
df2['Embarked'] = le.transform(df2['Embarked'])

# %% (cell 55)
df2.dropna(subset=['Age'], axis= 0, inplace = True)

# %% (cell 56)
df2.head()

# %% [markdown] (cell 57)
# ### Separating the traget and independent variable

# %% (cell 58)
x = df2.drop(columns=['Survived'])
y = df2['Survived']

# %% [markdown] (cell 59)
# ## Predicting using Decision Tree Classifier

# %% (cell 60)
tree_pred = dtree.predict(x)

# %% (cell 61)
from sklearn.metrics import accuracy_score
accuracy_score(y, tree_pred)

# %% [markdown] (cell 62)
# #### Confusion Matrix

# %% (cell 63)
from sklearn.metrics import confusion_matrix
sns.heatmap(confusion_matrix(y,tree_pred),annot= True, cmap = 'Blues')
plt.ylabel('Predicted Values')
plt.xlabel('Actual Values')
plt.title('confusion matrix')
plt.show()
