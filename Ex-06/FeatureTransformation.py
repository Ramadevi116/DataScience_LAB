# titanic_dataset.csv 
 
import pandas as pd   
import numpy as np
import matplotlib.pyplot as plt   
import seaborn as sns  
import statsmodels.api as sm   
import scipy.stats as stats   
 
df=pd.read_csv("titanic_dataset.csv")   
df   
 
df.drop("Name",axis=1,inplace=True)   
df.drop("Cabin",axis=1,inplace=True)  
df.drop("Ticket",axis=1,inplace=True)   
df.isnull().sum()   
 
df["Age"]=df["Age"].fillna(df["Age"].median())   
df["Embarked"]=df["Embarked"].fillna(df["Embarked"].mode()[0])   
df.info()   
 
from sklearn.preprocessing import OrdinalEncoder   
  
embark=["C","S","Q"]  
emb=OrdinalEncoder(categories=[embark])   
df["Embarked"]=emb.fit_transform(df[["Embarked"]])   
 
df   
 
#FUNCTION TRANSFORMATION:   
#Log Transformation   np.log(df["Fare"])   #ReciprocalTransformation   np.reciprocal(df["Age"])   
#Squareroot Transformation:   np.sqrt(df["Embarked"])   
 
#POWER TRANSFORMATION:   
df["Age _boxcox"], parameters=stats.boxcox(df["Age"])  
df   
df["Pclass _boxcox"], parameters=stats.boxcox(df["Pclass"])     
df     
df["Fare _yeojohnson"], parameters=stats.yeojohnson(df["Fare"])   
df   
df["SibSp _yeojohnson"], parameters=stats.yeojohnson(df["SibSp"])   
df   
df["Parch _yeojohnson"], parameters=stats.yeojohnson(df["Parch"])   
df   
 
#QUANTILE TRANSFORMATION   
from sklearn.preprocessing import QuantileTransformer    
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)   
df["Age_1"]=qt.fit_transform(df[["Age"]])   
sm.qqplot(df['Age'],line='45')   
plt.show()   
sm.qqplot(df['Age_1'],line='45')   
plt.show()   
df["Fare_1"]=qt.fit_transform(df[["Fare"]])   
sm.qqplot(df["Fare"],line='45')   
plt.show()   
sm.qqplot(df['Fare_1'],line='45')   
plt.show()   
 
df.skew()   
df  
# data_to_transform.csv 
 
import pandas as pd   
import numpy as np   
import matplotlib.pyplot as plt   
import seaborn as sns   
import statsmodels.api as sm   
import scipy.stats as stats   
df=pd.read_csv("Data_To_Transform.csv")   
df   
df.skew()   
 
#FUNCTION TRANSFORMATION:   
#Log Transformation   
 
np.log(df["Highly Positive Skew"])   
#Reciprocal Transformation   np.reciprocal(df["Moderate Positive Skew"])   
#Square Root Transformation   
np.sqrt(df["Highly Positive Skew"])   
#Square Transformation   
np.square(df["Highly Negative Skew"])   
 
#POWER TRANSFORMATION:   
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])   
df   
df["Moderate Positive Skew_yeojohnson"], parameters=stats.yeojohnson(df["Moderate Positive Skew"])   
df   
df["Moderate Negative Skew_yeojohnson"], parameters=stats.yeojohnson(df["Moderate Negative Skew"])   
df   
df["Highly Negative Skew_yeojohnson"], parameters=stats.yeojohnson(df["Highly Negative Skew"])   
df   
 
#QUANTILE TRANSFORMATION:   
from sklearn.preprocessing import QuantileTransformer    
qt=QuantileTransformer(output_distribution='normal')   
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])   
sm.qqplot(df['Moderate Negative Skew'],line='45')   
plt.show() 
sm.qqplot(df['Moderate Negative Skew_1'],line='45')   
plt.show()   
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])   
sm.qqplot(df['Highly Negative Skew'],line='45')   
plt.show()   
sm.qqplot(df['Highly Negative Skew_1'],line='45')  
plt.show()   
df["Moderate Positive Skew_1"]=qt.fit_transform(df[["Moderate Positive Skew"]])   
sm.qqplot(df['Moderate Positive Skew'],line='45')   
plt.show()   
sm.qqplot(df['Moderate Positive Skew_1'],line='45')   
plt.show()  
df["Highly Positive Skew_1"]=qt.fit_transform(df[["Highly Positive Skew"]])   
sm.qqplot(df['Highly Positive Skew'],line='45')   
plt.show()   
sm.qqplot(df['Highly Positive Skew_1'],line='45')   
plt.show()   
 
df.skew()   
df  
