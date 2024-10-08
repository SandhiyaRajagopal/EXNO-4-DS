# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/bmi.csv")
df.head()
```
![image](https://github.com/user-attachments/assets/544dd6ce-5bb7-4f02-af5d-acc4a145b308)
```
df.dropna()
```
![image](https://github.com/user-attachments/assets/43e4263a-93d9-499d-ac5e-56cd4b2a9d7f)
```
max_vals=np.max(np.abs(df[['Height','Weight']]))
max_vals
```
![image](https://github.com/user-attachments/assets/c59cd352-e683-41ee-b33f-b1fb2d830b0a)
```
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/user-attachments/assets/3c0f2dc7-5350-43a1-84a6-177584e9a3b2)
```
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/user-attachments/assets/84862e72-c4f5-47a8-b778-96024d00278b)
```
from sklearn.preprocessing import Normalizer
scaler=Normalizer()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```
![image](https://github.com/user-attachments/assets/7bb1edc5-8cbe-41f4-87ae-9115972f8189)
```
from sklearn.preprocessing import MaxAbsScaler
scaler=MaxAbsScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```
![image](https://github.com/user-attachments/assets/b72b66c7-7bd6-4342-a2d4-c4f0c90539b5)
```
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head()
```
![image](https://github.com/user-attachments/assets/9ffc4b19-bfaf-4a24-a73c-3218966eae93)
```
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
data=pd.read_csv('/content/income(1) (1).csv',na_values=[" ?"])
data
```
![image](https://github.com/user-attachments/assets/2d4b50a8-3bb6-4f2d-9677-45b508bec4ec)
```
data.isnull().sum()
```
![image](https://github.com/user-attachments/assets/72945a89-0cd5-4b8e-8db4-61774fa31347)
```
missing=data[data.isnull().any(axis=1)]
missing
```
![image](https://github.com/user-attachments/assets/854a52e5-2ead-4749-9323-363bd4760af9)
```
data2=data.dropna(axis=0)
data2
```
![image](https://github.com/user-attachments/assets/acb001c5-6a1d-447e-880a-a143c5427c85)
```
sal=data['SalStat']
data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![image](https://github.com/user-attachments/assets/d9bea70a-52e7-4e9b-b05a-0cee2d5779b2)
```
sal2=data['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![image](https://github.com/user-attachments/assets/0334155c-901a-44d5-bae9-4bbcb549f0b1)
```
data2
```
![image](https://github.com/user-attachments/assets/97885b39-c134-4c3b-bdaa-da1f4441f892)
```
new_data=pd.get_dummies(data2,drop_first=True)
new_data
```
![image](https://github.com/user-attachments/assets/01138704-dd86-42eb-969f-a087fd4d37d2)
```
columns_list=list(new_data.columns)
print(columns_list)
```
![image](https://github.com/user-attachments/assets/d21d5fb8-bb4f-4bd4-b5a4-0476b69d87a5)
```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![image](https://github.com/user-attachments/assets/27f23d8b-dbde-4ec0-a293-f485c9608f04)
```
y=new_data['SalStat'].values
print(y)
```
![image](https://github.com/user-attachments/assets/2ae366d7-f6fc-42fe-8df5-8e3c070b082e)
```
x=new_data[features].values
print(x)
```
![image](https://github.com/user-attachments/assets/24545a58-2d48-4da4-86ba-2331941c4eee)
```
x=new_data[features].values
print(x)
```
![image](https://github.com/user-attachments/assets/f1deeb48-e81d-4d8e-9df2-f613e94a1e0c)
```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors=5)
KNN_classifier.fit(train_x,train_y)
```
![image](https://github.com/user-attachments/assets/e6878688-d490-43b2-bea4-21ddd7004322)
```
prediction=KNN_classifier.predict(test_x)
confusionMmatrix=confusion_matrix(test_y,prediction)
print(confusionMmatrix)
```
![image](https://github.com/user-attachments/assets/4e68b7e4-b4d1-453b-bc62-794a9f667434)
```
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```
![image](https://github.com/user-attachments/assets/e4c537bc-a642-443a-a4aa-5a85144711b7)
```
print('Misclassified samples: %d' % (test_y != prediction).sum())
```
![image](https://github.com/user-attachments/assets/f9c870c2-feb3-42af-b6bb-5f49a446500e)
```
data.shape
```
![image](https://github.com/user-attachments/assets/330af299-783b-4eb1-80f6-cab487fcd34f)
```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns 
tips=sns.load_dataset('tips')
tips.head()
```
![image](https://github.com/user-attachments/assets/2c1d2cc5-59dc-4420-ba44-b8f830da94e4)
```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
![image](https://github.com/user-attachments/assets/1cdf5fcd-275f-4fb5-bf1c-ccb14753d929)
```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistic: {chi2}")
print(f"p-vale: {p}")
```
![image](https://github.com/user-attachments/assets/3b3aee59-235f-4acb-8233-643cb02a46d0)
```
import pandas as pd
from sklearn.feature_selection import SelectKBest,mutual_info_classif,f_classif
data={
    'Feature1':[1,2,3,4,5],
    'Feature2':['A','B','C','A','B'],
    'Feature3':[0,1,1,0,1],
    'Target':[0,1,1,0,1]
}
df=pd.DataFrame(data)
df
```
![image](https://github.com/user-attachments/assets/46bee2ce-7ad4-41f1-a1d1-731067fdd135)
```
X=df[['Feature1','Feature3']]
y=df['Target']
selector=SelectKBest(score_func=mutual_info_classif,k=1)
X_new=selector.fit_transform(X,y)
selected_features_indices=selector.get_support(indices=True)
selected_features=X.columns[selected_features_indices]
print("Selected Features:")
print(selected_features)
```
![image](https://github.com/user-attachments/assets/397882d8-acdb-4ba6-9e04-296235deeda4)

# RESULT:
       Thus the code executed successfully.
