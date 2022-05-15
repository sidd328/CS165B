# Starter code for CS 165B HW3

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


    # TODO: IMPLEMENT HERE

    pass


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

