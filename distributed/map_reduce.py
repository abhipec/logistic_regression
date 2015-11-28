import os
from mrjob.job import MRJob
import numpy as np
from scipy.stats import logistic 
number_of_examples = 100
learning_rate = 1

def sigmoid(value):
    return logistic.cdf(value)

def hypothesis(X,theta):
    return sigmoid(np.dot(theta,X))

class logistic_regression(MRJob):
    count = 0
    def mapper(self, _, line):
        self.count += 1
        sample = line.split(',')
        global theta
        if theta == "first_run":
            # initialize thetas to zeroes array with length equal to length of input
            theta = np.zeros(len(sample))
        sample = [1] + sample
        try:
            sample = list(map(float,sample))
            for j in range(len(theta)):
                value = (hypothesis(sample[:-1], theta) - sample[-1]) * sample[j]
                yield j, value
        except ValueError:
            pass

    def reducer(self, key, values):
        if isinstance(key, int):
            yield key, sum(values)
        elif key == "count":
            yield key, sum(values)
        else:
            yield key, max(values)
    
    def mapper_final(self):
        for j in range(len(theta)):
            yield 't' + str(j), theta[j]
        yield "count", self.count


def read_theta(path):
    if os.path.isfile(path):
        f = open(path)
        data = list(filter(None,f.read().split('\n')))
        theta = []
        for line in data:
            temp = line.split('\t')
            theta.append(float(temp[1]))
        return theta
    else:
        return None


if __name__ == '__main__':
    theta = read_theta('thetas.txt')
    if not theta:
        # this will make sure that number of coefficients are according to data
        theta = "first_run"
    logistic_regression.run()


