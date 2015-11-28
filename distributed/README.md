Algorithm : Regularized Logistic regression using gradient descent (map reduce implementation)

Programing Language : Python 2

Python libraries used : 
    1. Pandas for reading and manipulating data from csv file
    2. scipy for evaluation of sigmoid function
    3. numpy for matrix dot product
    4. Python MRjob for writing hadoop map reduce jobs in python

Code tested on Arch Linux and should work on any other operating system provided the dependent libraries are installed.

1. Python -> 3.5 
2. Scipy -> 0.16.1
3. Pandas -> 0.17.0
4. numpy -> 1.10
5. Mrjob -> 4.5
6. Hadoop must be installed and HADOOP_HOME should point to hadoop installation path

Execution instructions :
-l: learning rate
-i: number of iterations
-r: mrjob parameter

local testing
./logistic_distributed.zsh -f ../data/100.csv -l 1 -i 4 -r local

testing on hadoop
./logistic_distributed.zsh -f ../data/100.csv -l 1 -i 4 -r hadoop

output:
coefficients of hypothesis equation in file thetas

Data format :

Data file should contain values separated by ','.
Last column must be for labels .
There can be any number of columns or rows in data file.
First row will not be used for processing.


