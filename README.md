# Building, Breaking and Fixing a Neural Network

A complete deep learning study on **Fashion-MNIST** covering neural network implementation, backpropagation, activation functions, loss functions, optimization, overfitting, regularization, and hyperparameter tuning.

The project follows a controlled experimental approach: first building a neural network from scratch, then deliberately breaking it through overfitting, and finally improving generalization through regularization and hyperparameter selection.

---

## 📌 Project Overview

The purpose of this project is to understand the complete lifecycle of a neural network:

**Build → Verify → Compare → Overfit → Regularize → Tune → Evaluate**

The project includes seven major experimental parts:

1. Neural network and backpropagation from scratch
2. Activation function comparison
3. Loss function and regression experiments
4. Optimizer comparison and learning-rate tuning
5. Deliberate overfitting
6. Regularization and generalization experiments
7. Random hyperparameter search with 5-fold cross-validation

The final model was evaluated on the previously held-out Fashion-MNIST test set.

---

## 🎯 Objectives

The main objectives of the project were to:

* Implement a feedforward neural network using NumPy.
* Implement forward and backward propagation manually.
* Verify manually calculated gradients using PyTorch autograd.
* Compare different activation functions.
* Investigate vanishing gradients and dead ReLU neurons.
* Compare Cross-Entropy Loss and MSE for classification.
* Apply an MLP to a regression problem.
* Compare different optimization algorithms.
* Intentionally create an overfitted neural network.
* Evaluate multiple regularization techniques.
* Perform hyperparameter tuning using random search.
* Use 5-fold cross-validation for model selection.
* Evaluate the final model using accuracy, precision, recall, F1-score, and a confusion matrix.

---

# 📊 Dataset

## Fashion-MNIST

The main dataset used is **Fashion-MNIST**, which contains grayscale images of clothing and footwear.

| Dataset  | Samples | Image Size | Features |
| -------- | ------: | ---------: | -------: |
| Training |  60,000 |    28 × 28 |      784 |
| Test     |  10,000 |    28 × 28 |      784 |

Pixel values were normalized from:

```text
0–255
```

to:

```text
0–1
```

Each image was flattened into a 784-dimensional feature vector.

### Training / Validation Split

The 60,000 training samples were divided into:

| Split      | Samples |
| ---------- | ------: |
| Training   |  48,000 |
| Validation |  12,000 |
| Test       |  10,000 |

The provided test set was kept separate and used for final evaluation.

### Classes

| Label | Class       |
| ----: | ----------- |
|     0 | T-shirt/top |
|     1 | Trouser     |
|     2 | Pullover    |
|     3 | Dress       |
|     4 | Coat        |
|     5 | Sandal      |
|     6 | Shirt       |
|     7 | Sneaker     |
|     8 | Bag         |
|     9 | Ankle boot  |

---

# 🔬 Part 1 — Neural Network From Scratch

The first experiment implemented a feedforward neural network using **NumPy only**, without using a deep-learning framework for the actual implementation.

### Architecture

```text
784 Input Features
       ↓
64 Hidden Neurons
       ↓
     ReLU
       ↓
10 Output Neurons
       ↓
    Softmax
```

### Implemented Components

* Forward propagation
* ReLU activation
* Softmax
* Categorical Cross-Entropy
* Backpropagation
* Gradient calculation
* Gradient descent
* Weight updates

The model was trained on **5,000 samples for 20 epochs**.

### Results

| Metric              |     Result |
| ------------------- | ---------: |
| Training samples    |      5,000 |
| Epochs              |         20 |
| Final training loss |     0.4643 |
| Training accuracy   | **83.80%** |

### Gradient Check

The manually calculated gradients were compared with PyTorch autograd.

| Parameter | Maximum Absolute Difference |
| --------- | --------------------------: |
| W1        |                 1.49 × 10⁻⁸ |
| b1        |                 1.86 × 10⁻⁸ |
| W2        |                 2.24 × 10⁻⁸ |
| b2        |                 2.24 × 10⁻⁸ |

Overall maximum difference:

```text
2.24 × 10⁻⁸
```

This extremely small difference confirms that the manual backpropagation implementation was correct.

---

# 🧪 Part 2 — Activation Function Study

Four activation functions were compared using the same general network architecture and training setup.

### Architecture

```text
784 → 128 → 64 → 10
```

### Activation Functions

* Sigmoid
* Tanh
* ReLU
* Leaky ReLU

### Results

| Activation | Validation Accuracy | Validation Loss |
| ---------- | ------------------: | --------------: |
| Sigmoid    |              88.17% |          0.3318 |
| Tanh       |          **89.01%** |      **0.3087** |
| ReLU       |              86.76% |          0.3561 |
| Leaky ReLU |              87.89% |          0.3300 |

### Baseline

Tanh produced the best validation accuracy:

```text
89.01%
```

Therefore, the Tanh model was selected as the **Part 2 baseline**.

### Gradient Analysis

The mean absolute gradient of the first hidden layer was:

| Activation |  Epoch 1 | Final Epoch |
| ---------- | -------: | ----------: |
| Sigmoid    | 0.000164 |    0.000381 |
| ReLU       | 0.000497 |    0.000991 |

Approximately **52% of ReLU activations were zero** on the validation data.

This demonstrated the practical effects of gradient shrinkage with saturating activations and inactive ReLU units.

---

# 📉 Part 3 — Loss Functions and Regression

## Classification Loss Comparison

Cross-Entropy Loss and MSE with one-hot encoded targets were compared.

| Loss Function | Validation Accuracy |
| ------------- | ------------------: |
| Cross-Entropy |              87.92% |
| MSE           |          **88.06%** |

In this particular experiment, MSE produced a slightly higher validation accuracy.

## Regression Experiment

A separate MLP regression experiment was performed using the **California Housing dataset**.

### Architecture

```text
8 → 32 → 16 → 1
```

The input features were standardized before training.

### Results

| Metric | Result |
| ------ | -----: |
| MSE    | 0.3216 |
| RMSE   | 0.5671 |
| MAE    | 0.3913 |

This experiment demonstrated the use of neural networks for both classification and regression tasks.

---

# ⚙️ Part 4 — Optimizer Comparison

Four optimizers were evaluated:

* SGD
* SGD + Momentum
* RMSProp
* Adam

Multiple learning rates were tested for each optimizer.

### Learning-Rate Search

| Optimizer      | Learning Rates        |
| -------------- | ---------------------- |
| SGD            | 0.01, 0.05, 0.1       |
| SGD + Momentum | 0.01, 0.05, 0.1       |
| RMSProp        | 0.0001, 0.0005, 0.001 |
| Adam           | 0.0001, 0.0005, 0.001 |

### Best Optimizer Configuration

```text
Optimizer: RMSProp
Learning Rate: 0.001
Validation Accuracy: 88.48%
```

RMSProp with a learning rate of 0.001 achieved the strongest validation performance in this experiment.

---

# 🔥 Part 5 — Deliberate Overfitting

To study overfitting, the model capacity was intentionally increased while using only **2,000 training samples**.

### Overfit Architecture

```text
784
 ↓
512 + ReLU
 ↓
512 + ReLU
 ↓
512 + ReLU
 ↓
512 + ReLU
 ↓
10
```

The network contained approximately:

```text
1,195,018 trainable parameters
```

### Results

| Metric              |                      Result |
| ------------------- | --------------------------: |
| Training Accuracy   |                  **96.35%** |
| Validation Accuracy |                      80.59% |
| Generalization Gap  | **15.76 percentage points** |

The large difference between training and validation accuracy demonstrated clear **overfitting/high variance**.

The model learned the limited training data very well but did not generalize equally well to unseen validation examples.

---

# 🛠️ Part 6 — Regularization Experiments

Multiple techniques were evaluated to reduce the generalization gap.

The methods included:

* L1 regularization
* L2 regularization
* Dropout
* Batch normalization
* Early stopping
* Data augmentation
* Increasing the amount of training data

## Part 6 Summary Table

| Method              | Setting         | Train Accuracy | Validation Accuracy |         Gap |
| ------------------- | --------------- | -------------: | ------------------: | ----------: |
| Baseline            | —               |         96.35% |              80.59% |    15.76 pp |
| L2                  | λ = 0.0001      |         92.50% |              82.12% |    10.38 pp |
| L2                  | λ = 0.001       |         94.80% |              81.58% |    13.22 pp |
| L2                  | λ = 0.01        |         83.35% |              77.32% |     6.03 pp |
| L1                  | λ = 0.00001     |         95.20% |              78.94% |    16.26 pp |
| L1                  | λ = 0.00005     |         91.65% |              79.25% |    12.40 pp |
| L1                  | λ = 0.0001      |         90.40% |              81.23% |     9.17 pp |
| Dropout             | 0.2             |         92.85% |              81.23% |    11.62 pp |
| Dropout             | 0.4             |         88.75% |              83.02% |     5.72 pp |
| Dropout             | 0.6             |         82.05% |              81.78% | **0.27 pp** |
| Batch Normalization | —               |         98.15% |              79.66% |    18.49 pp |
| Early Stopping      | Patience = 5    |         88.10% |              81.42% |     6.68 pp |
| Data Augmentation   | Flip + Rotation |         87.70% |              82.08% |     5.62 pp |
| More Data           | 10,000 samples  |         94.72% |              85.19% |     9.53 pp |
| More Data           | 20,000 samples  |     **95.28%** |          **87.41%** |     7.88 pp |

### Best Practical Trade-off

Using **20,000 training samples** provided the best trade-off between reducing the generalization gap and preserving training accuracy.

```text
Training Accuracy:   95.28%
Validation Accuracy: 87.41%
Generalization Gap:   7.88 pp
```

Compared with the overfit baseline:

```text
Gap reduction:          7.88 pp
Training accuracy loss: 1.07 pp
```

Although dropout with a rate of 0.6 reduced the gap to only 0.27 pp, it also caused a much larger reduction in training accuracy. Therefore, 20,000 samples provided the better practical balance.

---

# 🎯 Part 7 — Hyperparameter Tuning

Random search was used to select a strong model configuration.

### Search Space

| Hyperparameter | Values                       |
| --------------- | ----------------------------- |
| Learning Rate   | 0.0001, 0.0005, 0.001, 0.005 |
| Batch Size      | 32, 64, 128                  |
| Dropout Rate    | 0.0, 0.2, 0.4, 0.5           |
| Hidden Size     | 128, 256, 512                |

A total of **12 random configurations** were evaluated using **5-fold cross-validation**.

### Selected Configuration

| Hyperparameter | Selected Value |
| --------------- | --------------: |
| Learning Rate   |     **0.0005** |
| Batch Size      |         **32** |
| Hidden Size     |        **512** |
| Dropout         |        **0.0** |

### Cross-Validation Result

```text
Mean CV Accuracy: 87.73%
CV Standard Deviation: 0.73%
```

---

# 🏆 Final Model Performance

The selected model was retrained on the full **48,000-sample training split** and evaluated once on the previously held-out Fashion-MNIST test set.

### Final Configuration

```text
Learning Rate = 0.0005
Batch Size    = 32
Hidden Size   = 512
Dropout       = 0.0
```

### Final Test Results

| Metric            |      Score |
| ----------------- | ---------: |
| **Test Accuracy** | **89.87%** |
| Macro Precision   | **0.9002** |
| Macro Recall      | **0.8987** |
| Macro F1-Score    | **0.8988** |

### Improvement

Part 2 baseline:

```text
89.01% validation accuracy
```

Final test accuracy:

```text
89.87%
```

Improvement:

```text
+0.86 percentage points
```

---

# 📊 Confusion Matrix

The final model produced the following confusion matrix:

```text
[[853,   0,  11,  26,   3,   0, 103,   0,   4,   0],
 [  3, 987,   1,   6,   1,   0,   1,   0,   1,   0],
 [ 26,   0, 767,  17, 118,   0,  70,   0,   2,   0],
 [ 13,  10,   4, 917,  35,   0,  20,   0,   1,   0],
 [  2,   1,  47,  15, 882,   0,  52,   0,   1,   0],
 [  0,   0,   0,   0,   0, 947,   1,  40,   1,  11],
 [101,   2,  49,  27,  65,   0, 754,   0,   2,   0],
 [  0,   0,   0,   0,   0,  10,   0, 968,   0,  22],
 [ 12,   0,   5,   1,   4,   2,   9,   2, 963,   2],
 [  1,   0,   0,   0,   0,   5,   0,  45,   0, 949]]
```

The model performed particularly well on visually distinctive classes such as **Trouser, Sandal, Sneaker, Bag, and Ankle boot**. More confusion occurred between visually similar clothing categories such as **Shirt, T-shirt/top, Pullover, and Coat**.

---

# 📋 Part 4 Summary Table

| Optimizer      |  Best Learning Rate |      Best Validation Accuracy |
| -------------- | -------------------: | ------------------------------: |
| SGD            |     0.01–0.1 tested | Lower than best configuration |
| SGD + Momentum |     0.01–0.1 tested | Lower than best configuration |
| **RMSProp**    |           **0.001** |                    **88.48%** |
| Adam           | 0.0001–0.001 tested | Lower than best configuration |

> The complete Part 4 table in the notebook includes the required optimizer, learning rate, epochs-to-85%-accuracy, final validation accuracy, and wall-clock time results.

---

# 📈 Required Plots

The notebook contains the required labelled plots for the experimental analysis, including:

* Part 1 training loss
* Part 2 activation-function validation loss curves
* Part 3 Cross-Entropy vs MSE training curves
* Part 4 optimizer training-loss curves
* Part 5 training vs validation loss
* Part 6 regularization/generalization-gap plots
* Part 7 final evaluation/confusion matrix

All plots are generated directly from the experimental results in the notebook.

---

# 🛠️ Technologies Used

* **Python**
* **NumPy**
* **Pandas**
* **Matplotlib**
* **Seaborn**
* **PyTorch**
* **Scikit-learn**
* **Google Colab / Jupyter Notebook**

---

# 🚀 Reproduction Instructions

## 1. Clone the Repository

```bash
git clone https://github.com/rimshabash/fashion-mnist-neural-network-from-scratch-to-tuning_Deep_Learning
cd Building-Breaking-Fixing-Neural-Network
```

## 2. Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn torch torchvision
```

## 3. Open the Notebook

The notebook can be opened using Jupyter:

```bash
jupyter notebook
```

or uploaded directly to **Google Colab**.

## 4. Prepare the Dataset

Place the Fashion-MNIST CSV files in the location expected by the notebook, or update the dataset paths in the first section of the notebook.

## 5. Run the Notebook

Run all notebook cells sequentially.

The notebook will:

1. Load and preprocess the dataset.
2. Create the training and validation split.
3. Train the NumPy neural network.
4. Perform gradient verification.
5. Compare activation functions.
6. Compare loss functions.
7. Run the regression experiment.
8. Compare optimizers.
9. Demonstrate deliberate overfitting.
10. Evaluate regularization techniques.
11. Perform random hyperparameter search.
12. Train the final model.
13. Evaluate the model on the held-out test set.
14. Generate the required plots and tables.

---

# ⚠️ Reproducibility Notes

* The Fashion-MNIST test set is kept separate from model-selection experiments and used for final evaluation.
* Random seeds should be kept fixed where specified in the notebook to improve reproducibility.
* Training time and exact runtime may vary depending on the available hardware.
* GPU acceleration is recommended for faster execution of the complete notebook.
* The final reported test score is **89.87%** from the completed experiment.

---

# 📚 Key Learning Outcomes

This project provided practical experience with:

* Neural network architecture
* Forward propagation
* Backpropagation
* Gradient descent
* Softmax classification
* Cross-Entropy Loss
* MSE
* Activation functions
* Vanishing gradients
* Dead ReLU neurons
* SGD
* Momentum
* RMSProp
* Adam
* Overfitting
* Bias and variance
* L1 regularization
* L2 regularization
* Dropout
* Batch normalization
* Early stopping
* Data augmentation
* Training-set size
* Random search
* K-fold cross-validation
* Classification metrics
* Confusion matrices
* Model generalization

---

# 🏁 Conclusion

This project demonstrated the complete process of **building, breaking, and fixing a neural network**.

The manually implemented neural network successfully passed gradient verification with a maximum difference of only **2.24 × 10⁻⁸**, confirming the correctness of the backpropagation implementation.

The overfitting experiment showed that a high-capacity model trained on limited data can achieve high training accuracy while performing substantially worse on unseen data. The regularization experiments demonstrated that different techniques provide different trade-offs between fitting the training data and improving generalization.

Random hyperparameter search with 5-fold cross-validation selected a model using a **learning rate of 0.0005, batch size 32, hidden size 512, and dropout 0.0**.

The final model achieved:

**89.87% test accuracy**
**0.9002 macro precision**
**0.8987 macro recall**
**0.8988 macro F1-score**

This represents an improvement of **0.86 percentage points** over the Part 2 baseline.

Overall, the project shows that successful neural-network development requires more than increasing model size: **careful experimentation, diagnosis of overfitting, appropriate regularization, and systematic hyperparameter selection are essential for good generalization.**

---

## 👩‍💻 Author

**Rimsha Bashir**
BS Computer Science
FAST NUCES, CFD Campus

---
