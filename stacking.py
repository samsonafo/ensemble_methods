#import libraries
import pandas as pd
from sklearn.ensemble import StackingClassifier,RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

#import  datasets
from sklearn.datasets import load_breast_cancer
X,y = load_breast_cancer(return_X_y=True)

#Train,Test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)

##preprocessing
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#logistic Regression
lr = LogisticRegression().fit(X_train,y_train)
y_predict = lr.predict(X_test) 
score_lr = f1_score(y_test,y_predict, average='micro')

print('The score of the Logistic Regression is {}'.format(score_lr))

#random Forest
rf = RandomForestClassifier(n_estimators=50).fit(X_train,y_train)
y_predict = rf.predict(X_test) 
score_rf = f1_score(y_test,y_predict, average='micro')

print('The score of the Random Forest is {}'.format(score_rf))

#nearest neigbours
nn = KNeighborsClassifier().fit(X_train,y_train)
y_predict = nn.predict(X_test) 
score_nn = f1_score(y_test,y_predict, average='micro')

print('The score of the Nearest Neigbours is {}'.format(score_nn))


#Stacking Classifier
estimators = [('lr',lr),('rf',rf),('nn',nn)]

clf = StackingClassifier(estimators=estimators,final_estimator=LogisticRegression())

clf.fit(X_train,y_train)

y_predict = clf.predict(X_test) 
score_clf = f1_score(y_test,y_predict, average='micro')
print('The score of the Stacking Classifier is {}'.format(score_clf))