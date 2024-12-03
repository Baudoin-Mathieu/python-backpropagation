import numpy as np
import csv

# sigmoid function
def sigmoid(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

def clean_print(var, msg):
    print(msg)
    print(var)
    print()
    return


# input dataset
X = np.empty((0,4), float)

# output dataset            
Y = np.empty((0,1), float)

# Populate datasets
f = open('eurusd_daily_2024.csv','rt')

f.readline() #Discard first line

csvFile = csv.reader(f)
for lines in csvFile:

    pClose = (float)(lines[1])
    pOpen = (float)(lines[2])
    pHigh = (float)(lines[3])
    pLow = (float)(lines[4])


    values = np.array([pClose, pOpen, pHigh, pLow])
    X = np.append(X, [values], axis=0)

    change = pClose - pOpen
    if(change > 0) :
        v = 1
    else :
        v = 0

    Y = np.append(Y, [[v]], axis=0)

f.close()
#---------------------

# Now we have to shift the Y space to 1 forward so that the value of the day n predict the change of day n+1

# Del 1st element
Y = np.delete(Y, 0, axis=0)

# Put a dummy at the last index so the dimension still match. the dummy will create error in the mesures but its fine 
Y = np.append(Y, [[1]], axis=0)



# print
print(X)
print("Dimension : ")
print(X.shape)
print()
print(Y)
print("Dimension : ")
print(Y.shape)
print()





# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# initialize weights randomly with mean 0
syn0 = 2*np.random.random((4,1)) - 1

clean_print(syn0,"Synapse 0 before Training")


iterNbr = (int)(input("Number of iterations : "))


for iter in range(iterNbr):

    # forward propagation
    l0 = X
    l1 = sigmoid(np.dot(l0,syn0))

    clean_print(l1, "l1 forward")

    # how much did we miss?
    l1_error = Y - l1

    clean_print(l1_error, "l1 error")

    # multiply how much we missed by the 
    # slope of the sigmoid at the values in l1
    l1_delta = l1_error * sigmoid(l1,True)

    clean_print(l1_delta, "l1 delta")

    # update weights
    syn0 += np.dot(l0.T,l1_delta)

    clean_print(syn0, "syn0")


clean_print(syn0,"Weights of synapse 0 After Training :")
clean_print(l1,"Final Output Dataset :")

