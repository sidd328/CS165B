# Starter code for CS 165B HW2 Spring 2022

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

    class1 = training_input[1:1+class1_num]
    class2 = training_input[1+class1_num:1+class1_num+class2_num]
    class3 = training_input[1+class1_num+class2_num:]

    centroidA = [sum(class1[:][i])/class1_num for i in range(class1_num)]
    centroidB = [sum(class2[:][i])/class1_num for i in range(class2_num)]
    centroidC = [sum(class3[:][i])/class1_num for i in range(class3_num)]

    
    


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

