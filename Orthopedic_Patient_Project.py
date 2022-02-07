#import libraries
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import export_text
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn import svm
from sklearn.metrics import classification_report
import seaborn as sns
import pandas as pd
import numpy as np


data_path = "column_3C_weka.csv"
#The dataset provides the patients’ information
data = pd.read_csv(data_path)
data.info(verbose = True)
print(data.describe().T)
#%%
# histogram
p = data.hist(figsize=(20,20))
print(p)

# scatter matrix
from pandas.plotting import scatter_matrix
p=scatter_matrix(data,figsize=(25, 25))

# Pair plot
p=sns.pairplot(data, hue = 'class')

#Correlation-heatmap
plt.figure(figsize=(12,10)) 
p=sns.heatmap(data.corr(), annot=True,cmap ='RdYlGn')

#%% K Nearest Neighbor Algorithm
labels = data["class"].values
features = data.drop("class",axis=1).values

#Train Test Split data==> 80% of data set for Train, 20% of data set for Test
features_train,features_test,labels_train,labels_test = train_test_split(features,labels,test_size=0.2,random_state=42,stratify=labels)

#Setup arrays to store train and test accuracies
neighbors = np.arange(1,11)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over different values of k
for i,k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors = k, metric= "manhattan")
    # Fit the classifier to the training data
    knn.fit(features_train,labels_train)
    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(features_train,labels_train)
    
    #Compute accuracy on the test set
    test_accuracy[i] = knn.score(features_test,labels_test)
    
# Generate plot
plt.title("Biomechanical features of orthopedic patients")
plt.plot(neighbors,train_accuracy,label="Train Accuracy")
plt.plot(neighbors,test_accuracy,label="Test Accuracy")
plt.legend()
plt.xlabel("Number Of Neighbors")
plt.ylabel("Accuracy")
plt.show()

#%%
# See details for best prediction
knn = KNeighborsClassifier(n_neighbors=8, metric= "manhattan")

#fitting and prediction
knn.fit(features_train,labels_train)
knn_pred = knn.predict(features_test)

#ACCURACY
print("KNN Test Accuracy Score",knn.score(features_test, labels_test))

#Model Performance Analysis

#let us get the predictions using the classifier we had fit above
labels_predict_KNN = knn.predict(features_test)

pd.crosstab(labels_test,labels_predict_KNN,rownames=["True"],colnames=["Predicted"],margins=True)

cnf_matrix = metrics.confusion_matrix(labels_test,labels_predict_KNN)
p = sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
plt.title('KNN Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


print("KNN Classification Report \n ",classification_report(labels_test,labels_predict_KNN))

#%% Support Vector Machines
kernel = ["linear","rbf","poly"]

for method in kernel:
    #create a classifier
    support_vector_machines = svm.SVC(kernel=method)
    #train the model
    support_vector_machines.fit(features_train,labels_train)
    #predict the response
    svm_pred = support_vector_machines.predict(features_test)
    #accuracy
    print("SVM {} Kernel Accuracy:".format(method), metrics.accuracy_score(labels_test,svm_pred))

# Best SVM kernel is polynomial
#create a classifier
support_vector_machines = svm.SVC(kernel="poly")
#train the model
support_vector_machines.fit(features_train,labels_train)
#predict the response
svm_pred = support_vector_machines.predict(features_test)

#accuracy
print("SVM Accuracy:", metrics.accuracy_score(labels_test,svm_pred))
print("SVM Classification Report \n ",metrics.classification_report(labels_test, svm_pred))

cnf_matrix_svm = metrics.confusion_matrix(labels_test,svm_pred)
p = sns.heatmap(pd.DataFrame(cnf_matrix_svm), annot=True, cmap="YlGnBu" ,fmt='g')
plt.title('SVM Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
#%% Decision Trees

#Setup arrays to store train and test accuracies
tree_depth = np.arange(1,7)
train_accuracy_trees = np.empty(len(tree_depth))
test_accuracy_trees = np.empty(len(tree_depth))

# Loop over different values of k
for i,k in enumerate(tree_depth):
    # Setup a k-NN Classifier with k neighbors: knn
    decision_trees = DecisionTreeClassifier(max_depth = k,random_state = 42)
    # Fit the classifier to the training data
    decision_trees.fit(features_train,labels_train)
    
    #Compute accuracy on the training set
    train_accuracy_trees[i] = decision_trees.score(features_train,labels_train)
    
    #Compute accuracy on the test set
    test_accuracy_trees[i] = decision_trees.score(features_test,labels_test)
    
# Generate plot
plt.title("Biomechanical features of orthopedic patients")
plt.plot(tree_depth,train_accuracy_trees,label="Decision Trees Train Accuracy")
plt.plot(tree_depth,test_accuracy_trees,label="Decision Trees Test Accuracy")
plt.legend()
plt.xlabel("Tree Depth")
plt.ylabel("Accuracy")
plt.show()

# Best Decision Tree has 2 as max_depth
decision_trees = DecisionTreeClassifier(max_depth =2, random_state = 42)
decision_trees.fit(features_train, labels_train)

features_dataframe = pd.DataFrame(features)
feature_names = features_dataframe.columns
#export the decision rules
tree_rules = export_text(decision_trees,
                        feature_names = list(feature_names))
#print the result
print("Tree Rules: ", tree_rules)

test_pred_decision_tree = decision_trees.predict(features_test)

print("Decision Trees Accuracy", metrics.accuracy_score(labels_test, test_pred_decision_tree))
print("Decision Trees Classification Report \n ",metrics.classification_report(labels_test, test_pred_decision_tree))

cnf_matrix_decision_trees = metrics.confusion_matrix(labels_test,test_pred_decision_tree)
p = sns.heatmap(pd.DataFrame(cnf_matrix_decision_trees), annot=True, cmap="YlGnBu" ,fmt='g')
plt.title('SVM Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')