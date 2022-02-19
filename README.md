# Assignment
### assignment 04 simple linear regration
1) Delivery_time -> Predict delivery time using sorting time 
2) Salary_hike -> Build a prediction model for Salary_hike

------------------------------------------------------------

Build a simple linear regression model by performing EDA and do necessary transformations and select the best model using R or Python.
1) Delivery_time -> Predict delivery time using sorting time¶
import pandas as pd 
import numpy as np
import seaborn as sns

# 1) Delivery_time -> Predict delivery time using sorting time 

data = pd.read_csv('delivery_time.csv')

data.head()

data.info()

# correlation



data.corr()

data1=data.rename({'Delivery Time':'delivery','Sorting Time':'sorting'},axis=1)

data1

import statsmodels.formula.api as smf
model = smf.ols("delivery~sorting",data = data1).fit()

sns.regplot(x = "sorting", y = "delivery",data = data1);

model.summary()

model.params

print( model.tvalues, '\n', model.pvalues)

(model.rsquared,model.rsquared_adj)

### predict for new data points

# predict for 11 and 12 sorting time

datas =pd.Series([11,12])

data_pred = pd.DataFrame(datas,columns = ['sorting'])

data_pred
predic = model.predict(data_pred)
print("PREDICTION VALUE :\n",predic)


H0 = y is not dependent on x
Ha = y is dependent on x
P value < alpha value there for the reject null hypothesis
the y is dependent on x
R-squared: 0.682



2) Salary_hike -> Build a prediction model for Salary_hike
data = pd.read_csv("Salary_Data.csv")

data.head()

data1 = data.rename({"YearsExperience":"Year"},axis = 1)

data1

data1.info()

# corelation

data1.corr()

sns.distplot(data1['Year'])

sns.distplot(data1['Salary'])

#### There is both the graphs are reprsent the positive skewness there for the the relation between Year and Salary is positive relation
#* x => independent varible => Year
#* y => dependent variable => salary

model = smf.ols("Salary~Year",data = data1).fit()

sns.regplot(x = "Year", y = "Salary", data = data1)

model.summary()

###  predict for new data points

## predict for 10 and 11 years of experince

newdata =pd.Series([10,11])

data_pre= pd.DataFrame(newdata,columns=['Year'])

data_pre

A = model.predict(data_pre)

print("PREDICTION VALUE : \n", A)
sns.regplot(x = "Year", y = "Salary", data = data1)
