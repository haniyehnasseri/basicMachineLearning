import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn import preprocessing
import random
from imblearn.over_sampling import RandomOverSampler
from sklearn import metrics
import datetime
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn import tree

le = preprocessing.LabelEncoder()
ros = RandomOverSampler(random_state=42)

    
data = pd.read_csv('data.csv')
dates = data['Date']
year = []
month = []
day = []
for date in dates:
    year.append(datetime.datetime.strptime(date, '%Y-%m-%d').year)
    month.append(datetime.datetime.strptime(date, '%Y-%m-%d').month)
    day.append(datetime.datetime.strptime(date, '%Y-%m-%d').day)
    

data['Year'] = year
data['Month'] = month
data['Day'] = day

x = data[['Total Quantity','Total Price','Country','Year','Month','Day','Purchase Count']]
y = data['Is Back']
y = le.fit_transform(y)
x = x.assign(Country=le.fit_transform(x['Country']))
#transformer = ColumnTransformer(transformers=[('cat', OneHotEncoder(), [2])])
#x = transformer.fit_transform(x)
#preprocessor = make_column_transformer( (OneHotEncoder(),['Country']),remainder="passthrough")
#x = preprocessor.fit_transform(x)
mi = mutual_info_classif(x,y,discrete_features=True)
print(mi)

## FAZ 1 ##
from sklearn import preprocessing
import csv
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import imblearn
from imblearn.over_sampling import SMOTE
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV



X = x
Y = y

#training_features,test_features,training_target,test_target, = train_test_split(X,Y,test_size = 0.1,random_state=12)
#X,Y = ros.fit_resample(X, Y)
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.15,random_state=12)
#sm = SMOTE()
#X_train_res, Y_train_res = sm.fit_sample(X_train, Y_train)
#X_train, Y_train = ros.fit_resample(X_train, Y_train)


# KNN #
#kfold = model_selection.KFold(n_splits = 15)
bestK = 0
bestacc = 0
for i in range(1,100):
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(X_train,Y_train)
    y_pred = model.predict(X_test)
    if(metrics.accuracy_score(Y_test, y_pred) > bestacc):
        bestacc = metrics.accuracy_score(Y_test, y_pred)
        bestK = i


print(bestK)
model = KNeighborsClassifier(n_neighbors=bestK)
model.fit(X_train,Y_train)
y_pred = model.predict(X_test)
print("Recall : ", metrics.recall_score(Y_test, y_pred))
print("Precision : ", metrics.precision_score(Y_test, y_pred))
print("Accuracy : ", metrics.accuracy_score(Y_test, y_pred))
#print(metrics.classification_report(Y_test, y_pred))
#print(model_selection.cross_val_score(model, X_test, Y_test, cv=kfold, scoring='precision').mean())
#print(model_selection.cross_val_score(model, X_test, Y_test, cv=kfold, scoring='recall').mean())
#print(model_selection.cross_val_score(model, X_test, Y_test, cv=kfold).mean())    
          
# Decision Tree #



    
bestDepth = 0
bestacc = 0
for i in range(1,100):
    clf = DecisionTreeClassifier(max_depth=i, random_state = 12)
    clf = clf.fit(X_train,Y_train)
    y_pred = clf.predict(X_test)
    if(metrics.accuracy_score(Y_test, y_pred) > bestacc):
        bestacc = metrics.accuracy_score(Y_test, y_pred)
        bestDepth = i



clf = DecisionTreeClassifier(max_depth=bestDepth, random_state = 12)
path = clf.cost_complexity_pruning_path(X_train, Y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities
clf = DecisionTreeClassifier(ccp_alpha=ccp_alphas[int(len(ccp_alphas) * 24/25)], random_state=12)
clf = clf.fit(X_train,Y_train)
y_pred = clf.predict(X_test)
print("Recall : ", metrics.recall_score(Y_test, y_pred))
print("Precision : ", metrics.precision_score(Y_test, y_pred))
print("Accuracy : ", metrics.accuracy_score(Y_test, y_pred))


# Logistic #

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(solver='newton-cg', max_iter=10000)
logreg.fit(X_train,Y_train)
y_pred=logreg.predict(X_test)
print("Recall : ", metrics.recall_score(Y_test, y_pred))
print("Precision : ", metrics.precision_score(Y_test, y_pred))
print("Accuracy : ", metrics.accuracy_score(Y_test, y_pred))


### Questions of Faz 1 ###

#1)

print("Compare between Test and Train data: ")

# calculated for test data in previous parts #

# KKN #


y_pred = model.predict(X_train)
print("Recall : ", metrics.recall_score(Y_train, y_pred))
print("Precision : ", metrics.precision_score(Y_train, y_pred))
print("Accuracy : ", metrics.accuracy_score(Y_train, y_pred))
          
          
# Decision Tree #

y_pred = clf.predict(X_train)
print("Recall : ", metrics.recall_score(Y_train, y_pred))
print("Precision : ", metrics.precision_score(Y_train, y_pred))
print("Accuracy : ", metrics.accuracy_score(Y_train, y_pred))


# Logistic #

y_pred=logreg.predict(X_train)
print("Recall : ", metrics.recall_score(Y_train, y_pred))
print("Precision : ", metrics.precision_score(Y_train, y_pred))
print("Accuracy : ", metrics.accuracy_score(Y_train, y_pred))



#2)

# knn plot

def knnExamplePlot():
    
    neighbors = np.arange(1, 50) 
    train_accuracy = np.empty(len(neighbors)) 
    test_accuracy = np.empty(len(neighbors)) 
  

    for i, k in enumerate(neighbors): 
        knn = KNeighborsClassifier(n_neighbors=k) 
        knn.fit(X_train, Y_train) 
        train_accuracy[i] = knn.score(X_train, Y_train)
        test_accuracy[i] = knn.score(X_test, Y_test) 

    plt.plot(neighbors, test_accuracy, label = 'Testing dataset Accuracy') 
    plt.plot(neighbors, train_accuracy, label = 'Training dataset Accuracy') 
  
    plt.legend() 
    plt.xlabel('n_neighbors') 
    plt.ylabel('Accuracy') 
    plt.show()


knnExamplePlot()

# Decision Tree plot

def decisionTreeExamplePlot():
    
    depths = np.arange(1,20) 
    train_accuracy = np.empty(len(depths)) 
    test_accuracy = np.empty(len(depths)) 
  

    for i, k in enumerate(depths): 
        dt = DecisionTreeClassifier(max_depth=k, random_state = 12)
        path = dt.cost_complexity_pruning_path(X_train, Y_train)
        ccp_alphas, impurities = path.ccp_alphas, path.impurities
        dt = DecisionTreeClassifier(ccp_alpha=ccp_alphas[int(len(ccp_alphas) * 24/25)],random_state=12)
        dt.fit(X_train, Y_train)
        train_accuracy[i] = dt.score(X_train, Y_train) 
        test_accuracy[i] = dt.score(X_test, Y_test) 
  
    plt.plot(depths, test_accuracy, label = 'Decision Tree Testing dataset Accuracy') 
    plt.plot(depths, train_accuracy, label = 'Decision Tree Training dataset Accuracy') 
  
    plt.legend() 
    plt.xlabel('Max Depth') 
    plt.ylabel('Accuracy') 
    plt.show()


decisionTreeExamplePlot()

## Faz 2 ##

def bagging(baseClassifier):
    seed = 10
    #kfold = model_selection.KFold(n_splits = 10)
    num = 20
    if(baseClassifier == "knn"):
        
        bagging = BaggingClassifier(base_estimator = model, max_samples=0.5, max_features=0.5, n_estimators = num, random_state = 12)
        #bagging = BaggingClassifier(tree.DecisionTreeClassifier(random_state=1))
        bagging.fit(X_train, Y_train)
        print("Test Results : ")
        print("KNN Score:\t", model.score(X_test,Y_test))
        print("Bagging Score:\t", bagging.score(X_test,Y_test))
        print("Train Results : ")
        print("KNN Score:\t", model.score(X_train,Y_train))
        print("Bagging Score:\t", bagging.score(X_train,Y_train))
        print("\n\n")
        #print("KNN Score:\t", metrics.accuracy_score(Y_test, model.predict(X_test)))
        #print("Bagging Score:\t", metrics.accuracy_score(Y_test, bagging.predict(X_test)))
        #print("KNN Score:\t", model_selection.cross_val_score(model, X_test, Y_test, cv=kfold).mean())
        #print("Bagging Score:\t", model_selection.cross_val_score(bagging, X_test, Y_test, cv=kfold).mean())
        return bagging

    else:
        bagging = BaggingClassifier(base_estimator = clf, max_samples=0.5, max_features=0.5, n_estimators = num,random_state = 12)
        #bagging = BaggingClassifier(tree.DecisionTreeClassifier(random_state=12),max_samples=0.5, max_features=0.5)
        bagging.fit(X_train, Y_train)
        print("Test Results : ")
        print("Tree Score:\t", clf.score(X_test,Y_test))
        print("Bagging Score:\t", bagging.score(X_test,Y_test))
        print("Train Results : ")
        print("Tree Score:\t", clf.score(X_train,Y_train))
        print("Bagging Score:\t", bagging.score(X_train,Y_train))
        print("\n\n")
        #print("Tree Score:\t", metrics.accuracy_score(Y_test, clf.predict(X_test)))
        #print("Bagging Score:\t", metrics.accuracy_score(Y_test, bagging.predict(X_test)))
        #print("Tree Score:\t", model_selection.cross_val_score(clf, X_test, Y_test, cv=kfold).mean())
        #print("Bagging Score:\t", model_selection.cross_val_score(bagging, X_test, Y_test, cv=kfold).mean())
        return bagging
        
#1)

knnBagging = bagging("knn")
decisionTreeBagging = bagging("decision tree")

#2)

def checkSamplesonForest():

    #kfold = model_selection.KFold(n_splits = 10)
    
    estimators = np.arange(1,25) 
    test_accuracy_random = np.empty(len(estimators))
    train_accuracy_random = np.empty(len(estimators))
    #test_accuracy_tree = np.empty(len(estimators)) 
  

    for i, k in enumerate(estimators): 
        rforest = RandomForestClassifier(n_estimators=k, random_state = 12)
        rforest.fit(X_train, Y_train)

        #decisionTree = DecisionTreeClassifier(max_depth=k)
        #decisionTree.fit(X_train, Y_train) 
        #train_accuracy_random[i] = model_selection.cross_val_score(rforest, X_train, Y_train, cv=kfold).mean()
        #test_accuracy_random[i] = model_selection.cross_val_score(rforest, X_test, Y_test, cv=kfold).mean()
        train_accuracy_random[i] = rforest.score(X_train, Y_train)
        #test_accuracy_tree[i] = decisionTree.score(X_test, Y_test)
        test_accuracy_random[i] = rforest.score(X_test, Y_test) 
  
    #plt.plot(estimators, test_accuracy_tree, label = 'Tree test Accuracy')
    plt.plot(estimators, test_accuracy_random, label = 'Forest test Accuracy - Samples') 
    plt.plot(estimators, train_accuracy_random, label = 'Forest train Accuracy - Samples') 
  
    plt.legend() 
    plt.xlabel('Number of Estimators') 
    plt.ylabel('Accuracy')
    plt.show()



def checkMaxDepthForest():

    #kfold = model_selection.KFold(n_splits = 10)
    
    depths = np.arange(1,25) 
    test_accuracy_random = np.empty(len(depths))
    train_accuracy_random = np.empty(len(depths))
    #test_accuracy_tree = np.empty(len(estimators)) 
  

    for i, k in enumerate(depths): 
        rforest = RandomForestClassifier(max_depth=k, random_state = 12)
        rforest.fit(X_train, Y_train)

        #decisionTree = DecisionTreeClassifier(max_depth=k)
        #decisionTree.fit(X_train, Y_train)

        #train_accuracy_random[i] = model_selection.cross_val_score(rforest, X_train, Y_train, cv=kfold).mean()
        #test_accuracy_random[i] = model_selection.cross_val_score(rforest, X_test, Y_test, cv=kfold).mean()
      
        train_accuracy_random[i] = rforest.score(X_train, Y_train)
        #test_accuracy_tree[i] = decisionTree.score(X_test, Y_test)
        test_accuracy_random[i] = rforest.score(X_test, Y_test) 
  
    #plt.plot(estimators, test_accuracy_tree, label = 'Tree test Accuracy')
    plt.plot(depths, test_accuracy_random, label = 'Forest test Accuracy - Depth') 
    plt.plot(depths, train_accuracy_random, label = 'Forest train Accuracy - Depth') 
  
    plt.legend() 
    plt.xlabel('Max Depth') 
    plt.ylabel('Accuracy')
    plt.show()


def checkMaxLeafForest():
    #kfold = model_selection.KFold(n_splits = 10)
    
    leaves = np.arange(2,25) 
    test_accuracy_random = np.empty(len(leaves))
    train_accuracy_random = np.empty(len(leaves))
    #test_accuracy_tree = np.empty(len(estimators)) 
  

    for i, k in enumerate(leaves): 
        rforest = RandomForestClassifier(max_leaf_nodes=k,random_state=12)
        rforest.fit(X_train, Y_train)

        #decisionTree = DecisionTreeClassifier(max_depth=k)
        #decisionTree.fit(X_train, Y_train)

        #train_accuracy_random[i] = model_selection.cross_val_score(rforest, X_train, Y_train, cv=kfold).mean()
        #test_accuracy_random[i] = model_selection.cross_val_score(rforest, X_test, Y_test, cv=kfold).mean()
      
        train_accuracy_random[i] = rforest.score(X_train, Y_train)
        test_accuracy_random[i] = rforest.score(X_test, Y_test)
        #test_accuracy_random[i] = rforest.score(X_test, Y_test) 
  
    #plt.plot(estimators, test_accuracy_tree, label = 'Tree test Accuracy')
    plt.plot(leaves, test_accuracy_random, label = 'Forest test Accuracy - leaves') 
    plt.plot(leaves, train_accuracy_random, label = 'Forest train Accuracy - leaves') 
  
    plt.legend() 
    plt.xlabel('Max Leaf Nodes') 
    plt.ylabel('Accuracy')
    plt.show()
    

def randomForest(clf):
    #kfold = model_selection.KFold(n_splits = 10)
    rforest=RandomForestClassifier(random_state=12)
    distributions = dict(n_estimators=[i for i in range(1,25)],max_leaf_nodes=[i for i in range(1,25)])
    rforest = RandomizedSearchCV(rforest, distributions, random_state=12)
    rforest.fit(X_train,Y_train)
    print("Test Results : ")
    print("Tree Score:\t", clf.score(X_test,Y_test))
    print("Forest Score:\t", rforest.score(X_test,Y_test))
    print("Train Results : ")
    print("Tree Score:\t", clf.score(X_train,Y_train))
    print("Forest Score:\t", rforest.score(X_train,Y_train))
    #y_pred=rforest.predict(X_test)
    #print("Forest Acc:\t", model_selection.cross_val_score(rforest, X_test, Y_test, cv=kfold).mean())
    #print("Forest precision:\t", model_selection.cross_val_score(rforest, X_test, Y_test, cv=kfold, scoring='precision').mean())
    #print("Forest recall:\t", model_selection.cross_val_score(rforest, X_test, Y_test, cv=kfold, scoring='recall').mean())
    #print("Random Forest Classification Report:")
    #print(metrics.classification_report(Y_test, y_pred))
    #print("Recall : ", metrics.recall_score(Y_test, y_pred, average=None).mean())
    #print("Precision : ", metrics.precision_score(Y_test, y_pred, average=None).mean())
    #print("Accuracy : ", metrics.accuracy_score(Y_test, y_pred))


    

    

randomForest(clf)

checkSamplesonForest()

#checkMaxDepthForest()

checkMaxLeafForest()

#3)

def checkBaggingEffectOnOverFitting_decisionTree():
    
    depths = np.arange(1,50) 
    train_accuracy = np.empty(len(depths)) 
    test_accuracy = np.empty(len(depths)) 

    
    num = 20

    for i, k in enumerate(depths):
        knn = KNeighborsClassifier(n_neighbors=k)
        bagging = BaggingClassifier(base_estimator = knn, max_samples=0.5, max_features=0.5,n_estimators = num,random_state = 12)
        bagging.fit(X_train, Y_train) 
        train_accuracy[i] = bagging.score(X_train, Y_train)
        test_accuracy[i] = bagging.score(X_test, Y_test) 


    plt.plot(depths, test_accuracy, label = 'Testing dataset Accuracy - Bagging Overfit') 
    plt.plot(depths, train_accuracy, label = 'Training dataset Accuracy - Bagging Overfit') 
  
    plt.legend() 
    plt.xlabel('Max Depth') 
    plt.ylabel('Accuracy') 
    plt.show()


    

checkBaggingEffectOnOverFitting_decisionTree()
knnExamplePlot()


#4) Hard Voting

hard_voting_clf = VotingClassifier(
    estimators=[('model', model), ('clf', clf), ('logreg', logreg)], 
    voting='hard')
hard_voting_clf.fit(X_train, Y_train)
y_pred = hard_voting_clf.predict(X_test)
print("Recall : ", metrics.recall_score(Y_test, y_pred))
print("Precision : ", metrics.precision_score(Y_test, y_pred))
print("Accuracy : ", metrics.accuracy_score(Y_test, y_pred))



