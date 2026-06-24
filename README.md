# Neural Network From Scratch (XOR Solver)

A clean, dependency-free implementation of a Multi-Layer Perceptron (MLP) built entirely from scratch using Python and NumPy. 

This repository serves as an educational breakdown of foundational deep learning concepts, demonstrating how a network utilizes **forward propagation**, **Mean Squared Error (MSE)**, and **backpropagation via the chain rule** to solve the classic non-linear XOR (Exclusive OR) problem.

---

## 📐 The XOR Problem

The XOR gate is a fundamental problem in computational complexity. Because the outputs are not linearly separable, a single-layer perceptron without a hidden layer cannot converge on a solution. This implementation introduces a $2 \times 4 \times 1$ network architecture to successfully map the non-linear decision boundary.

| Input $X_1$ | Input $X_2$ | Target Output $Y$ |
| :---: | :---: | :---: |
| 0 | 0 | **0** |
| 0 | 1 | **1** |
| 1 | 0 | **1** |
| 1 | 1 | **0** |

---

## 🧠 Network Architecture & Mathematics

The network maps a 2-dimensional input space through a 4-dimensional hidden representation to output a single scalar probability between 0 and 1.



### 1. Forward Propagation
Data flows sequentially from left to right through matrix multiplication, bias addition, and element-wise activation:

#### **Hidden Layer**
The input matrix is projected into a 4-dimensional hidden space:
$$z_1 = X \cdot W_1 + b_1$$
$$a_1 = \sigma(z_1)$$

> **Dimensions:** $X \in \mathbb{R}^{4 \times 2}$, $W_1 \in \mathbb{R}^{2 \times 4}$, and $b_1 \in \mathbb{R}^{1 \times 4}$

#### **Output Layer**
The hidden representations are compressed into a single scalar prediction:
$$z_2 = a_1 \cdot W_2 + b_2$$
$$a_2 = \sigma(z_2)$$

> **Dimensions:** $a_1 \in \mathbb{R}^{4 \times 4}$, $W_2 \in \mathbb{R}^{4 \times 1}$, and $b_2 \in \mathbb{R}^{1 \times 1}$

### 2. Activation Function
We use the **Sigmoid** activation function to map arbitrary real-valued inputs into a probability distribution curve bounded by $(0, 1)$:

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

To calculate gradients during the backward pass, we compute its first derivative:

$$\sigma'(x) = \sigma(x) \cdot (1 - \sigma(x))$$

### 3. Loss Optimization
The network measures structural variance via **Mean Squared Error (MSE)**:

$$L = \frac{1}{n} \sum_{i=1}^{n} (a_2^{(i)} - y^{(i)})^2$$

---

## 📉 Backpropagation & Gradient Descent

To minimize $L$, we compute the partial derivatives of the cost function with respect to every weight and bias parameter using the **Calculus Chain Rule**.



### Layer 2 Error Gradients
$$\delta_2 = \frac{\partial L}{\partial z_2} = \frac{\partial L}{\partial a_2} \odot \sigma'(z_2) = \frac{2}{n}(a_2 - y) \odot (a_2 \odot (1 - a_2))$$

$$\frac{\partial L}{\partial W_2} = a_1^T \cdot \delta_2, \quad \frac{\partial L}{\partial b_2} = \sum \delta_2$$

### Layer 1 Error Gradients
$$\delta_1 = \frac{\partial L}{\partial z_1} = (\delta_2 \cdot W_2^T) \odot \sigma'(z_1)$$

$$\frac{\partial L}{\partial W_1} = X^T \cdot \delta_1, \quad \frac{\partial L}{\partial b_1} = \sum \delta_1$$

### Parameter Updates
Parameters are adjusted dynamically in the opposite direction of the gradient step scaled by the learning rate ($\alpha$):

$$W_i \leftarrow W_i - \alpha \frac{\partial L}{\partial W_i}, \quad b_i \leftarrow b_i - \alpha \frac{\partial L}{\partial b_i}$$

---

## ⚡ Technical Setup

### Prerequisites
The environment requires standard Python 3.x along with `numpy` for vectorized matrix mathematics.

```bash
pip install numpy
