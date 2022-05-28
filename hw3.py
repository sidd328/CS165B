# Starter code for CS 165B HW3
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.metrics import zero_one_loss

def run_train_test(training_file, testing_file):
    """
    Implement the training and testing procedure here. You are permitted
    to use additional functions but DO NOT change this function definition. 

    Inputs:
        training_file: file object returned by open('training.txt', 'r')
        testing_file: file object returned by open('test1/2/3.txt', 'r')

    Output:
        Dictionary of result values 

        IMPORTANT: YOU MUST USE THE SAME DICTIONARY KEYS SPECIFIED
        
        Example:
            return {
    			"gini":{
    				'True positives':0, 
    				'True negatives':0, 
    				'False positives':0, 
    				'False negatives':0, 
    				'Error rate':0.00
    				},
    			"entropy":{
    				'True positives':0, 
    				'True negatives':0, 
    				'False positives':0, 
    				'False negatives':0, 
    				'Error rate':0.00}
    				}
    """


    #Read Data
    df = pd.read_csv(training_file, delim_whitespace=True)
    df = df.drop(columns=['#'])
    testdf = pd.read_csv(testing_file, delim_whitespace=True)
    testdf.drop(columns=['#'])

    #Seperate Features from class labels
    training_data = df[['Budget', 'Genre', 'FamousActors', 'Director']]
    training_labels = df[['GoodMovie']]
    testing_data = testdf[['Budget', 'Genre', 'FamousActors', 'Director']]
    testing_labels = testdf[['GoodMovie']]

    #Convert to arrays
    training_data = training_data.to_numpy()
    training_labels = training_labels.to_numpy().reshape(1, training_labels.shape[0])[0]
    testing_data = testing_data.to_numpy()
    testing_labels = testing_labels.to_numpy().reshape(1, testing_labels.shape[0])[0]
    
    #Predict using Decision Tree for Gini Index
    clf = tree.DecisionTreeClassifier(criterion='gini', random_state=0)
    clf = clf.fit(training_data, training_labels)
    pred_gini = clf.predict(testing_data)

    #Predict using Decision Tree for Entropy
    clf = tree.DecisionTreeClassifier(criterion='entropy', random_state=0)
    clf = clf.fit(training_data, training_labels)
    pred_entropy = clf.predict(testing_data)

    #Create Confusion matrix
    confusion_gini = np.zeros((2,2))
    confusion_entropy = np.zeros((2,2))

    for i in range(len(testing_labels)):
        if(testing_labels[i] == 0):
            if(pred_gini[i] == 0):
                confusion_gini[1,1] = confusion_gini[1,1] + 1
            else:
                confusion_gini[0,1] = confusion_gini[0,1] + 1

            if(pred_entropy[i] == 0):
                confusion_entropy[1,1] = confusion_entropy[1,1] + 1
            else:
                confusion_entropy[0,1] = confusion_entropy[0,1] + 1
        else:
            if(pred_gini[i] == 1):
                confusion_gini[0,0] = confusion_gini[0,0] + 1
            else:
                confusion_gini[1,0] = confusion_gini[1,0] + 1

            if(pred_entropy[i] == 1):
                confusion_entropy[0,0] = confusion_entropy[0,0] + 1
            else:
                confusion_entropy[1,0] = confusion_entropy[1,0] + 1

    #Return required values
    return {
    			"gini":{
    				'True positives':confusion_gini[0,0], 
    				'True negatives':confusion_gini[1,1], 
    				'False positives':confusion_gini[0,1], 
    				'False negatives':confusion_gini[1,0], 
    				'Error rate':(confusion_gini[1,0] + confusion_gini[0,1])/len(testing_labels)
    				},
    			"entropy":{
    				'True positives':confusion_entropy[0,0], 
    				'True negatives':confusion_entropy[1,1], 
    				'False positives':confusion_entropy[0,1], 
    				'False negatives':confusion_entropy[1,0], 
    				'Error rate':(confusion_entropy[1,0] + confusion_entropy[0,1])/len(testing_labels)
                }
    }


#######
# The following functions are provided for you to test your classifier.
#######

if __name__ == "__main__":
    """
    You can use this to test your code.
    python hw3.py [training file path] [testing file path]
    """
    import sys

    training_file = open(sys.argv[1], "r")
    testing_file = open(sys.argv[2], "r")

    run_train_test(training_file, testing_file)

    training_file.close()
    testing_file.close()

