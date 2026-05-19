Small fully connected neural network in NumPy on MNIST handwritten-digit images so it can predict which digit (0–9) each 28×28 image shows, and it reports how accurate those predictions are on handwritten digits (0-9).

* NumPy-only MLP with one hidden layer that classifies MNIST digits with around 90% or higher test accuracy

High-level Training Flow
1. Load MNIST images and labels from mnist.npz
2. Normalize pixel values from [0, 255] to [0, 1]
3. Flatten each 28 x 28 img into a 784 dimensional vector
4. Shuffle the dataset
5. Split into training and held-out test data
6. Convert labels to one-hot vectors
7. Intialize the MLP weights and biases
8. Train for multiple epochs using mini-batch gradient descent
9. Print loss, training accuract, and test accuracy for each epoch
10. Report final accuracy
