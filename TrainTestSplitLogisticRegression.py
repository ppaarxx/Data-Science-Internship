import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
df = pd.read_csv("C:/Users/parth puri/PycharmProjects/datascience/dataset1/IRIS.csv")
# print (df)

Logr= LogisticRegression()
X= df.drop("species",axis=1)
# print(X)
Y= df["species"]
# print(Y)

X_train,X_test,Y_train,Y_test= train_test_split(X,Y,random_state=0,test_size=0.3)
print (X_train)
print (X_test)
print (Y_train)
print (Y_test)

train = Logr.fit(X_train,Y_train)
Y_pred=Logr.predict(X_test)
print(Y_pred,Y_test)
print(accuracy_score(Y_test,Y_pred))