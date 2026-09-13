from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error,accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd

data_set=pd.DataFrame({
    "size_sqft":[1000,1200,1500,1800,2000,2200,2500,2800,3000,3500],
    "bedrooms":[2,3,3,4,4,4,5,5,5,6],
    "age_years":[10,8,5,6,4,3,2,1,1,0],
    "Price":[120000,150000,190000,230000,260000,290000,340000,380000, 410000,480000]
})

# pandas insertion

print(data_set.head(4))
data_set.info()
print(data_set.describe())
print(data_set.isna())
print(data_set.isna().sum())
print(data_set.shape)

# model of linear regression 

X1=data_set[["size_sqft","bedrooms","age_years"]]
Y1=data_set[["Price"]]

X1_train ,X1_test , Y1_train, Y1_test=train_test_split(X1,Y1,random_state=10,test_size=0.2)
model1=LinearRegression()

model1.fit(X1_train,Y1_train)
prediction_linear=model1.predict(X1_test)
print(prediction_linear)

mse=mean_squared_error(Y1_test,prediction_linear)
print(mse)

print(model1.coef_)
print(model1.intercept_)


# model of the logistics regression

X2=data_set[["size_sqft","bedrooms","age_years"]]
data_set["result"]=(data_set["Price"]>300000).astype(int)
Y2=data_set["result"]

X2_train ,X2_test , Y2_train, Y2_test=train_test_split(X2,Y2,random_state=10,test_size=0.2)
model2=LogisticRegression()

model2.fit(X2_train,Y2_train)
prediction_logistics=model2.predict(X2_test)
print(prediction_logistics)

probability=model2.predict_proba(X2_test)
print(probability)

accuracy=accuracy_score(Y2_test,prediction_logistics)
print(accuracy)


print(model2.coef_)
print(model2.intercept_)