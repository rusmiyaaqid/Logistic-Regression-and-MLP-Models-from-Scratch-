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
    parser = argparse.ArgumentParser(description="MLP with one hidden layer (10 units), batch gradient descent")
    parser.add_argument("file", type=str, help="Input file")
    parser.add_argument("alpha", type=float, help="Learning rate")
    parser.add_argument("N", type=int, help="Number of iterations")
    args = parser.parse_args()

    file = args.file
    alpha = args.alpha
    N = args.N

    # Load data
    y = []
    X = []
    with open(file, "r") as f:
        for line in f:
            parts = line.strip().split(" ")
            x_vals = [float(val) for val in parts[:-1]]
            X.append(x_vals)
            y.append(float(parts[-1]))

    input_dim = len(X[0])
    hidden_units = 10

    hidden_w, hidden_b, output_w, output_b = initialize_mlp(input_dim, hidden_units)

    # Prepare CSV file for writing accuracy per epoch
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"1_mlp_accuracy_{file}_{alpha}_{N}_{timestamp}.csv"

    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Epoch", "Accuracy"])

        for epoch in range(N):
            total_loss = 0
            correct = 0

            for i in range(len(X)):
                _, _, _, output_a = forward_pass(X[i], hidden_w, hidden_b, output_w, output_b)
                total_loss += cross_entropy(y[i], output_a)
                pred = 1 if output_a >= 0.5 else 0
                if pred == y[i]:
                    correct += 1

            hidden_w, hidden_b, output_w, output_b = batch_backward_pass(
                X, y, hidden_w, hidden_b, output_w, output_b, alpha
            )

            acc = correct / len(X)
            writer.writerow([epoch + 1, acc])

            #if (epoch + 1) % 100 == 0 or epoch == N - 1:
            #    print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}, Accuracy = {acc:.4f}")

    print("\nFinal weights:")
    print("Hidden weights:", hidden_w)
    print("Hidden biases:", hidden_b)
    print("Output weights:", output_w)
    print("Output bias:", output_b)

if __name__ == "__main__":
    main()