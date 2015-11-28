import os
import numpy as np
import argparse

#Initialize argument parser
parser = argparse.ArgumentParser()
parser.add_argument("-l", "--learning_rate", type=float, help=    "learning rate")
args = parser.parse_args()
# default, changed by reading count in output
number_of_examples = 0

if args.learning_rate:
    learning_rate = args.learning_rate
else:
    learning_rate = 1
    print('Using default learning rate', learning_rate)


def save_theta(theta, path):
    f = open(path, 'w')
    for i in range(len(theta)):
        f.write(str(i) + '\t' + str(theta[i]) + '\n')
    f.close()


def update_theta():
    #read theta diff from output directory
    theta_error_dict = {}
    theta_dict = {}
    global number_of_examples
    f = open('output')
    data = list(filter(None,f.read().split('\n')))
    for line in data:
        temp = line.split('\t')
        if temp[0][0] == '"':
            if temp[0] == '"count"':
                number_of_examples = int(temp[1]) - 1
            else:
                # this will be value of theta
                theta_dict[int(temp[0][2:].replace('"',''))] = float(temp[1])
        else:
            theta_error_dict[int(temp[0])] = float(temp[1])  
    f.close()

    theta_prev = []
    for i in range(len(theta_dict)):
        theta_prev.append(theta_dict[i])

    theta_diff = []
    for i in range(len(theta_error_dict.keys())):
        # regularization on theta
        summition = theta_error_dict[i]
        if i != 0:
            summition -= theta_prev[i]/number_of_examples
        theta_diff.append(learning_rate * summition / number_of_examples)
    theta_prev = np.array(theta_prev) - np.array(theta_diff)
    save_theta(theta_prev, 'thetas.txt')

update_theta()
print(number_of_examples)

