#arobbin7@u.rochester.edu

import math
import argparse
import random
import csv
from datetime import datetime

def dot_product(x, y):
    return sum([x[i] * y[i] for i in range(len(x))])

#return the sigmoid of x
def sigmoid(x):
    return 1 / (1 + (math.e) ** (-x))
    

#return yhat
def logistic_regression(theta, x, bias):
    
    return sigmoid(dot_product(theta, x) + bias)

#cross entropy loss function
def cross_entropy(y, yhat):
    epsilon = 1e-15
    yhat = max(min(yhat, 1 - epsilon), epsilon)
    return y * math.log(yhat) + (1-y) * math.log(1 - yhat)

def gradient_descent_update(theta, alpha, y, x, N, bias):
    for i in range(0, len(theta)):
        sum =0
        for j in range(len(y)):
            yhat = logistic_regression(theta, x[j], bias)
            error = yhat - y[j]
            sum += error * x[j][i]
        theta[i] -= alpha * sum / len(y)
    #update the bias
    sum = 0
    for j in range(len(y)):
        yhat = logistic_regression(theta, x[j], bias)
        error = yhat - y[j]
        sum += error
    bias -= alpha * sum / len(y)
    return theta, bias



    


def main():
    #Parse command line
    
    parser = argparse.ArgumentParser(description="Logistic Regression with Gradient Descent")
    parser.add_argument("file", type=str, help="File name")
    parser.add_argument("alpha", type=float, help="Learning rate")
    parser.add_argument("N", type=int, help="Number of iterations")
    args = parser.parse_args()

    file = args.file
    alpha = args.alpha
    N = args.N
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"avgErrors_{file}_{alpha}_{N}_{timestamp}.csv"

    y =[]
    x=[]
    for line in open(file, "r"):
        #parse the line
        line = line.strip().split(" ")
        xTemp =[0] * 4
        
        for i in range(len(line)-1):
            xTemp[i] = float(line[i])
        y.append(float(line[-1]))
        x.append(xTemp)
    bias = random.uniform(0, 1)
    theta = [random.uniform(0, 1) for i in range(len(xTemp))]
    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Iteration', 'Average Error'])
        for i in range(N):
            theta, bias = gradient_descent_update(theta, alpha, y, x, N, bias)
            errorSum = 0
            for j in range(len(y)):
                errorSum = errorSum + cross_entropy(y[j], logistic_regression(theta, x[j], bias))
            #print("Average Error: ", errorSum / len(y))
            writer.writerow([i+1, abs(errorSum / len(y))])
    #for i in range(len(x)):
        #error = cross_entropy(y[i], logistic_regression(theta, x[i], bias))
        #print(logistic_regression(theta, x[i], bias))
        #print("Error: ", error)
    for i in range(len(theta)):
        print(theta[i], end = " ")
    print(bias)                                                                            

if __name__ == "__main__":
    main()