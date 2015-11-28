Algorithm : Regularized Logistic regression using gradient descent (parallel implementation)

Programing Language : Python 3

Python libraries used : 
    1. Pandas for reading and manipulating data from csv file
    2. scipy for evaluation of sigmoid function
    3. numpy for matrix dot product
    4. multiprocessing for shared memory parallel execution
Code tested on Arch Linux and should work on any other operating system provided the dependent libraries are installed.

1. Python -> 3.5 
2. Scipy -> 0.16.1
3. Pandas -> 0.17.0
4. numpy -> 1.10

Execution instructions :

python logistic_parallel.py -f ../data/100.csv -l 1 -i 100
or
python logistic_parallel.py --filename ../data/100.csv --learning_rate 1 --iterations 100

output:
coefficients of hypothesis equation 

Data format :

Data file should contain values separated by ','.
Last column must be for labels .
There can be any number of columns or rows in data file.
First row will not be used for processing.


