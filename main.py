import math
import argparse
import random
import csv
from datetime import datetime

def dot_product(x, y):
    return sum([x[i] * y[i] for i in range(len(x))])

def sigmoid(x):
    return 1 / (1 + (math.e) ** (-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def cross_entropy(y, yhat):
    epsilon = 1e-15
    yhat = max(min(yhat, 1 - epsilon), epsilon)
    return -(y * math.log(yhat) + (1 - y) * math.log(1 - yhat))

#logistic reg:

def logistic_regression(theta, x, bias):
    return sigmoid(dot_product(theta, x) + bias)

def gradient_descent_update(theta, alpha, y, X, bias):
    n_samples = len(y)

    # Update theta
    for i in range(len(theta)):
        grad = 0
        for j in range(n_samples):
            yhat = logistic_regression(theta, X[j], bias)
            grad += (yhat - y[j]) * X[j][i]
        theta[i] -= alpha * grad / n_samples

    # Update bias
    grad_b = sum([(logistic_regression(theta, X[j], bias) - y[j]) for j in range(n_samples)])
    bias -= alpha * grad_b / n_samples

    return theta, bias

def calculate_accuracy_logreg(theta, X, y, bias):
    correct = 0
    for i in range(len(y)):
        yhat = logistic_regression(theta, X[i], bias)
        pred = 1 if yhat >= 0.5 else 0
        if pred == y[i]:
            correct += 1
    return correct / len(y)
#MLP

def initialize_mlp(n_inputs, n_hidden):

    hidden_weights = []
    for i in range(n_hidden): 
        weights = []
        for _ in range(n_inputs):  
            weights.append(random.uniform(-1, 1))
        hidden_weights.append(weights)
    
    hidden_biases = []
    for i in range(n_hidden):
        hidden_biases.append(random.uniform(-1, 1))
    
   
    output_weights = []
    for i in range(n_hidden):
        output_weights.append(random.uniform(-1, 1))
        
    output_bias = random.uniform(-1, 1)
    return hidden_weights, hidden_biases, output_weights, output_bias

def forward_pass(x, hidden_weights, hidden_biases, output_weights, output_bias):
    hidden_activations = []
    hidden_zs = []
    for i in range(len(hidden_weights)):
        z = dot_product(hidden_weights[i], x) + hidden_biases[i]
        a = sigmoid(z)
        hidden_zs.append(z)
        hidden_activations.append(a)
    output_z = dot_product(output_weights, hidden_activations) + output_bias
    output_a = sigmoid(output_z)
    return hidden_activations, hidden_zs, output_z, output_a

def batch_backward_pass(X, y, hidden_weights, hidden_biases, output_weights, output_bias, lr):
    n_samples = len(X)

    d_output_w = [0.0] * len(output_weights)
    d_output_b = 0.0
    d_hidden_w = [[0.0 for _ in range(len(X[0]))] for _ in range(len(hidden_weights))]
    d_hidden_b = [0.0] * len(hidden_biases)

    for i in range(n_samples):
        x_i = X[i]
        y_i = y[i]

        h_acts, h_zs, o_z, o_a = forward_pass(x_i, hidden_weights, hidden_biases, output_weights, output_bias)
        dL_dz2 = (o_a - y_i) * sigmoid_derivative(o_z)

        # Gradients for output layer
        for j in range(len(output_weights)):
            d_output_w[j] += dL_dz2 * h_acts[j]
        d_output_b += dL_dz2

        # Gradients for hidden layer
        for j in range(len(hidden_weights)):
            dz1 = sigmoid_derivative(h_zs[j]) * dL_dz2 * output_weights[j]
            for k in range(len(hidden_weights[j])):
                d_hidden_w[j][k] += dz1 * x_i[k]
            d_hidden_b[j] += dz1

    # since we do batch gradient descent, average gradients before updating weights (we've already added all the gradients)
    for j in range(len(output_weights)):
        output_weights[j] -= lr * d_output_w[j] / n_samples
    output_bias -= lr * d_output_b / n_samples

    for j in range(len(hidden_weights)):
        for k in range(len(hidden_weights[j])):
            hidden_weights[j][k] -= lr * d_hidden_w[j][k] / n_samples
        hidden_biases[j] -= lr * d_hidden_b[j] / n_samples

    return hidden_weights, hidden_biases, output_weights, output_bias


def main():
    parser = argparse.ArgumentParser(description="Run Logistic Regression or MLP")
    parser.add_argument("--model", choices=["logreg", "mlp"], required=True,
                        help="Choose model: logreg or mlp")
    parser.add_argument("file", type=str, help="Input data file")
    parser.add_argument("alpha", type=float, help="Learning rate")
    parser.add_argument("N", type=int, help="Number of iterations")
    args = parser.parse_args()

    # Load Data
    X = []
    y = []
    with open(args.file, "r") as f:
        for line in f:
            parts = line.strip().split(" ")
            X.append([float(p) for p in parts[:-1]])
            y.append(float(parts[-1]))

    input_dim = len(X[0])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.model == "logreg":
        csv_name = f"logreg_accuracy_{args.file}_{args.alpha}_{args.N}_{timestamp}.csv"
        theta = [random.uniform(0,1) for _ in range(input_dim)]
        bias = random.uniform(0,1)

        with open(csv_name, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["Iteration", "Accuracy"])

            for i in range(args.N):
                theta, bias = gradient_descent_update(theta, args.alpha, y, X, bias)
                acc = calculate_accuracy_logreg(theta, X, y, bias)
                writer.writerow([i + 1, acc])

        print("Final theta:", theta)
        print("Final bias:", bias)

    elif args.model == "mlp":
        csv_name = f"mlp_accuracy_{args.file}_{args.alpha}_{args.N}_{timestamp}.csv"
        hidden_w, hidden_b, output_w, output_b = initialize_mlp(input_dim, 10)

        with open(csv_name, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Accuracy"])

            for epoch in range(args.N):
                correct = 0
                for i in range(len(X)):
                    _, _, _, out = forward_pass(X[i], hidden_w, hidden_b, output_w, output_b)
                    pred = 1 if out >= 0.5 else 0
                    if pred == y[i]:
                        correct += 1

                hidden_w, hidden_b, output_w, output_b = batch_backward_pass(
                    X, y, hidden_w, hidden_b, output_w, output_b, args.alpha
                )

                acc = correct / len(X)
                writer.writerow([epoch + 1, acc])

        print("Final Output Weights:", output_w)
        print("Output Bias:", output_b)


if __name__ == "__main__":
    main()