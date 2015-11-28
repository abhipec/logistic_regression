from multiprocessing import Process, Manager
import pandas as pd
from scipy.stats import logistic
import numpy as np
import argparse
import matplotlib.pyplot as plt

#Initialize argument parser
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filename", help="input file name")
parser.add_argument("-l", "--learning_rate", type=float, help="learning rate")
parser.add_argument("-i", "--iterations", type=int, help="number of iteration")
args = parser.parse_args()

if args.filename:
    filename = args.filename
else:
    print('Please specify a file name')
    print('Execute "python logistic.py -h" for help')
    exit()

if args.learning_rate:
    learning_rate = args.learning_rate
else:
    learning_rate = 1
    print('Using default learning rate', learning_rate)

if args.iterations:
    iterations = args.iterations
else:
    iterations = 100
    print('Using default iterations', iterations)

def compute_sum(parameters, result):
    for parameter in parameters:
        summition = 0
        for sample in data_array:
            summition += (hypothesis(sample[:-1], theta) - sample[-1]) * sample[parameter] 
        if parameter != 0:
            summition -= theta[parameter]/number_of_examples
        result[parameter] = learning_rate * summition / number_of_examples


def sigmoid(value):
    return logistic.cdf(value)

def hypothesis(X, theta):
    hx = np.dot(theta, X)
    return sigmoid(hx)

def predict_proba(X,theta):
    return sigmoid(np.inner(theta[1:], X) + theta[0])

#read data from csv file
data_csv = pd.read_csv(filename)

#number of training examples  
number_of_examples = data_csv.shape[0]

#number of parameters in training set  
number_of_parameters = data_csv.shape[1]

#add column of ones to dataframe  make looping symmetric 
ones = np.ones(number_of_examples)
data_csv.insert(0, 'theta0',ones)

#initialize theta to zeroes array
theta = np.zeros(number_of_parameters)

#convert dataframe to numpy array
data_array = data_csv.as_matrix()

parameters_split = np.array_split(list(range(len(theta))), 4)
print(parameters_split)
manager = Manager()
results = manager.dict()


for i in range(iterations):
    processes = []
    for j in range(len(parameters_split)):
        processes.append(Process(target=compute_sum, args=(parameters_split[j], results)))
    theta_diff = []
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    for j in range(len(theta)):
        theta_diff.append(results[j])
    theta = theta - theta_diff

for i in range(len(theta)):
    print('coefficient for theta ' + str(i), theta[i])

if len(data_csv.columns.tolist()) == 4:
    # plot decision boundary 
    # create 4 by 4 grid
    xx, yy = np.mgrid[-4:4:.01, -4:4:.01]
    grid = np.c_[xx.ravel(), yy.ravel()]

    # calculate probabilities of grid points 
    probs = predict_proba(grid, theta).reshape(xx.shape)

    # plot probability labels 
    f, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(xx, yy, probs, levels=[0.499, 0.501], cmap="Greys",vmin=0, vmax=0.6)

    # plot actual points on dataset 
    plt.title('Parallel')
    ax.scatter(data_csv['P1'], data_csv['P2'], c=data_csv['OUT'], s=50,cmap="RdBu", vmin=-.2, vmax=1.2,edgecolor="white", linewidth=1)
    ax.set(aspect="equal",xlim=(-4, 4), ylim=(-4, 4),xlabel="$X_1$", ylabel="$X_2$")
    plt.show()


