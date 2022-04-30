# Starter code for CS 165B HW2 Spring 2022
from matplotlib import testing
import numpy as np

def run_train_test(training_input, testing_input):
    """
    Implement the training and testing procedure here. You are permitted
    to use additional functions but DO NOT change this function definition. 
    You are permitted to use the numpy library but you must write 
    your own code for the linear classifier. 

    Inputs:
        training_input: list form of the training file
            e.g. [[3, 5, 5, 5],[.3, .1, .4],[.3, .2, .1]...]
        testing_input: list form of the testing file

    Output:
        Dictionary of result values 

        IMPORTANT: YOU MUST USE THE SAME DICTIONARY KEYS SPECIFIED
        
        Example:
            return {
                "tpr": #your_true_positive_rate,
                "fpr": #your_false_positive_rate,
                "error_rate": #your_error_rate,
                "accuracy": #your_accuracy,
                "precision": #your_precision
            }
    """


    #Get number of features and points of each class
    num_features = training_input[0][0]
    class1_num = training_input[0][1]
    class2_num = training_input[0][2]
    class3_num = training_input[0][3]

    #Get data points for each class
    class1 = np.array(training_input[1:1+class1_num])
    class2 = np.array(training_input[1+class1_num:1+class1_num+class2_num])
    class3 = np.array(training_input[1+class1_num+class2_num:])
    
    #Calculate centroids
    centroidA = class1.sum(axis=0)/class1_num
    centroidB = class2.sum(axis=0)/class2_num
    centroidC = class3.sum(axis=0)/class3_num

    
    #Calculate the parameters (normal and boundary value) for each boundary
    normalAB = (centroidA - centroidB)
    midpointAB = (centroidA + centroidB)/2
    tAB = np.dot(normalAB, midpointAB)
    
    normalBC = (centroidB - centroidC)
    midpointBC = (centroidB + centroidC)/2
    tBC = np.dot(normalBC, midpointBC)

    normalAC = (centroidA - centroidC)
    midpointAC = (centroidA + centroidC)/2
    tAC = np.dot(normalAC, midpointAC)


    #Classify testing points
    testing_tot = sum(testing_input[0][1:])
    result = []
    for i in range (1, testing_tot+1):
        point = testing_input[i]
        if (np.dot(point, normalAB) - tAB >= 0):
            if (np.dot(point, normalAC) - tAC >= 0):
                result.append("A")
            else:
                result.append("C")
        else:
            if (np.dot(point, normalBC) - tBC >= 0):
                result.append("B")
            else:
                result.append("C")


    #Confusion matrix
    confusion = np.zeros((3,3))

    #Get true values
    trueA = testing_input[0][1]
    trueB = testing_input[0][2]
    trueC = testing_input[0][3]

    #Get predicted values and fill in confusion matrix
    predA = predB = predC = 0
    for i in range(len(result)):
        if(result[i] == "A"):
            predA = predA + 1
            if(i < trueA):
                confusion[0][0] = confusion[0][0] + 1
            elif(trueA <= i < (trueB + trueA)):
                confusion[0][1] = confusion[0][1] + 1
            else:
                confusion[0][2] = confusion[0][2] + 1
        elif(result[i] == "B"):
            predB = predB + 1
            if(i < trueA):
                confusion[1][0] = confusion[1][0] + 1
            elif(trueA <= i < (trueB + trueA)):
                confusion[1][1] = confusion[1][1] + 1
            else:
                confusion[1][2] = confusion[1][2] + 1
        else:
            predC = predC + 1
            if(i < trueA):
                confusion[2][0] = confusion[2][0] + 1
            elif(trueA <= i < (trueB + trueA)):
                confusion[2][1] = confusion[2][1] + 1
            else:
                confusion[2][2] = confusion[2][2] + 1

    

    #Calculate results 
    #TPR
    TPR = (confusion[0][0]/trueA + confusion[1][1]/trueB + confusion[2][2]/trueC)/3

    #FPR
    FPRA = (confusion[0][1] + confusion[0][2])/(trueB + trueC)
    FPRB = (confusion[1][0] + confusion[1][2])/(trueA + trueC)
    FPRC = (confusion[2][0] + confusion[2][1])/(trueA + trueB)
    FPR = (FPRA + FPRB + FPRC)/3

    #Accuracy
    accuracyA = (confusion[0][0] + confusion[1][1] + confusion[1][2] + confusion[2][1] + confusion[2][2])/testing_tot
    accuracyB = (confusion[1][1] + confusion[0][0] + confusion[0][2] + confusion[2][0] + confusion[2][2])/testing_tot
    accuracyC = (confusion[2][2] + confusion[0][0] + confusion[0][1] + confusion[1][0] + confusion[1][1])/testing_tot
    accuracy = (accuracyA + accuracyB + accuracyC)/3

    #Error
    error = 1 - accuracy

    #Precision
    Precision = (confusion[0][0]/predA + confusion[1][1]/predB + confusion[2][2]/predC)/3

    return {
        "tpr": TPR,
        "fpr": FPR,
        "error_rate": error,
        "accuracy": accuracy,
        "precision": Precision
    }
    

#######
# The following functions are provided for you to test your classifier.
######
def parse_file(filename):
    """
    This function is provided to you as an example of the preprocessing we do
    prior to calling run_train_test
    """
    with open(filename, "r") as f:
        data = [[float(y) for y in x.strip().split(" ")] for x in f]
        data[0] = [int(x) for x in data[0]]

        return data

if __name__ == "__main__":
    """
    You can use this to test your code.
    python hw2.py [training file path] [testing file path]
    """
    import sys

    training_input = parse_file(sys.argv[1])
    testing_input = parse_file(sys.argv[2])

    run_train_test(training_input, testing_input)

