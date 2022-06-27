import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier


df = pd.read_csv("C:/Users/parth puri/PycharmProjects/datascience/dataset1/IRIS.csv")
# print (df)

### lOGISTIC REGRESSION

Logr= LogisticRegression()
Loga = RandomForestClassifier()
Logb = GradientBoostingClassifier()
Logc = MultinomialNB()
Logd = DecisionTreeClassifier()
Loge = svm.SVC()
Logf = MLPClassifier()
X= df.drop("species",axis=1)
# print(X)
Y= df["species"]
# print(Y)
logf = MLPClassifier(solver= 'lbfgs', alpha =1e-5,hidden_layer_sizes= (8,2), random_state = 0)

X_train,X_test,Y_train,Y_test= train_test_split(X,Y,random_state=0,test_size=0.3)
# print (X_train)
# print (X_test)
# print (Y_train)
# print (Y_test)

train = Logr.fit(X_train,Y_train)
Y_pred=Logr.predict(X_test)
# print(Y_pred,Y_test)
print("Logistic regression",accuracy_score(Y_test,Y_pred))

train = Loga.fit(X_train,Y_train)
Y_pred=Loga.predict(X_test)
# print(Y_pred,Y_test)
print("RandomForestClassifier",accuracy_score(Y_test,Y_pred))

train = Logb.fit(X_train,Y_train)
Y_pred=Logb.predict(X_test)
# print(Y_pred,Y_test)
print("GradientBoostingClassifier",accuracy_score(Y_test,Y_pred))

train = Logc.fit(X_train,Y_train)
Y_pred=Logc.predict(X_test)
# print(Y_pred,Y_test)
print("MultinomialNB",accuracy_score(Y_test,Y_pred))

train = Logd.fit(X_train,Y_train)
Y_pred=Logd.predict(X_test)
# print(Y_pred,Y_test)
print("DecisionTreeClassifier",accuracy_score(Y_test,Y_pred))

train = Loge.fit(X_train,Y_train)
Y_pred=Loge.predict(X_test)
# print(Y_pred,Y_test)
print("svm.SVC",accuracy_score(Y_test,Y_pred))

train = Logf.fit(X_train,Y_train)
Y_pred=Logf.predict(X_test)
# print(Y_pred,Y_test)
print("MLPClassifier",accuracy_score(Y_test,Y_pred))