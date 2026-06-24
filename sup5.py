import numpy as np

# =============================================================
# NEURAL NETWORK FROM SCRATCH — no tensorflow, no pytorch
# We'll teach it XOR: the simplest problem a linear model CAN'T solve
#
#   Input A | Input B | Output
#     0     |    0    |   0
#     0     |    1    |   1
#     1     |    0    |   1
#     1     |    1    |   0
#
# Structure: 2 inputs → 4 hidden neurons → 1 output
# =============================================================


# ── ACTIVATION FUNCTION ────────────────────────────────────────
# Sigmoid "squashes" any number into a value between 0 and 1.
# This is what makes neurons fire (close to 1) or not (close to 0).
#
#   sigmoid(x) = 1 / (1 + e^-x)
#
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# The DERIVATIVE of sigmoid — needed during backpropagation.
# If s = sigmoid(x), then sigmoid'(x) = s * (1 - s)
# It tells us: "how sensitive was the output to a small change in input?"
def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


# ── TRAINING DATA ──────────────────────────────────────────────
# XOR truth table. Each row is one example.
X = np.array([
    [0, 0],   # → 0
    [0, 1],   # → 1
    [1, 0],   # → 1
    [1, 1],   # → 0
])

y = np.array([[0], [1], [1], [0]])   # correct answers (column vector)


# ── WEIGHTS & BIASES ───────────────────────────────────────────
# Weights are the "knobs" the network adjusts while learning.
# We initialise them randomly (small values, not zeros — zeros would
# make all neurons learn the same thing).
#
# Layer 1: connects 2 inputs → 4 hidden neurons  (shape 2×4)
# Layer 2: connects 4 hidden → 1 output neuron   (shape 4×1)
#
np.random.seed(1)    # so you get the same result each run

W1 = np.random.randn(2, 4) * 0.5   # weights, input → hidden
b1 = np.zeros((1, 4))              # bias for hidden layer

W2 = np.random.randn(4, 1) * 0.5   # weights, hidden → output
b2 = np.zeros((1, 1))              # bias for output layer


# ── HYPERPARAMETERS ────────────────────────────────────────────
lr     = 2.0    # learning rate: how big each update step is
epochs = 10000  # how many times we loop over the full dataset


# ── TRAINING LOOP ──────────────────────────────────────────────
print("Training...\n")

for epoch in range(epochs):

    # ── FORWARD PASS ──────────────────────────────────────────
    # Data flows LEFT → RIGHT through the network.
    # Each layer: multiply inputs by weights, add bias, apply sigmoid.

    # Hidden layer
    z1 = X @ W1 + b1        # (4,2)·(2,4) + (1,4) → shape (4,4)
    a1 = sigmoid(z1)        # apply activation → hidden layer output

    # Output layer
    z2 = a1 @ W2 + b2       # (4,4)·(4,1) + (1,1) → shape (4,1)
    a2 = sigmoid(z2)        # final prediction, values between 0–1

    # ── LOSS (how wrong are we?) ───────────────────────────────
    # Mean Squared Error: average of (prediction - truth)²
    loss = np.mean((a2 - y) ** 2)

    # ── BACKWARD PASS (backpropagation) ───────────────────────
    # Data flows RIGHT → LEFT. We calculate how much each weight
    # contributed to the error, then nudge it in the right direction.

    # --- Output layer gradients ---
    # How much did the output error change w.r.t. z2?
    dL_da2 = 2 * (a2 - y) / len(y)          # derivative of MSE
    da2_dz2 = sigmoid_derivative(z2)         # derivative of sigmoid
    delta2 = dL_da2 * da2_dz2               # chain rule

    # How much to update W2 and b2?
    dW2 = a1.T @ delta2                      # (4,4).T · (4,1) → (4,1)
    db2 = np.sum(delta2, axis=0, keepdims=True)

    # --- Hidden layer gradients ---
    # Propagate the error back through W2 into the hidden layer
    da1_dz1 = sigmoid_derivative(z1)
    delta1 = (delta2 @ W2.T) * da1_dz1     # (4,1)·(1,4) * (4,4)

    dW1 = X.T @ delta1                      # (2,4).T · (4,4) → (2,4)
    db1 = np.sum(delta1, axis=0, keepdims=True)

    # ── GRADIENT DESCENT UPDATE ───────────────────────────────
    # Subtract the gradient scaled by learning rate.
    # Moving AGAINST the gradient reduces the loss.
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

    # Print progress every 1000 epochs
    if epoch % 1000 == 0:
        print(f"Epoch {epoch:5d}  |  Loss: {loss:.6f}")


# ── RESULTS ────────────────────────────────────────────────────
print("\n--- Final Predictions ---")
print(f"{'Input':<12} {'Predicted':>10} {'Correct':>10} {'Result':>8}")
print("-" * 45)

for i in range(len(X)):
    pred = a2[i][0]
    correct = y[i][0]
    label = "✓" if round(pred) == correct else "✗"
    print(f"{str(X[i]):<12} {pred:>10.4f} {int(correct):>10}    {label}")

print(f"\nFinal Loss: {loss:.6f}")
print("(Loss near 0 = model learned perfectly)")