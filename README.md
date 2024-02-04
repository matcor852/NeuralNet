### **NeuralNet**

This project is an implementation of a feedforward neural network in C99, focusing on online supervised learning, and providing a polyvalent framework for customization. The neural network supports 37 activation functions, 15 cost functions, and 7 predefined optimizers, making it a versatile tool for various machine learning tasks. Users can easily extend the functionality by adding their own activation functions, cost functions, and optimizers.

### Features

- **Feedforward Neural Network:**
  - Implementation of a neural network capable of handling feedforward architectures
  
- **Online Learning:**
  - The network processes one training example at a time, facilitating online learning scenarios

- **Supervised Learning:**
  - Designed for supervised learning tasks, where the model learns from labeled training data

- **Polyvalent Activation Functions:**
  - Offers a choice of 37 activation functions, providing flexibility for different model architectures

- **Customizable Cost Functions:**
  - Supports 15 cost functions, and users can easily define and integrate their own cost functions

- **Versatile Optimizers:**
  - Seven predefined optimizers (Adam, RMSProp, Nadam, Adamax, Momentum, Adagrad and Adadelta), allowing users to choose the optimization strategy that suits their needs

- **Regularization Support:**
  - Includes support for L1 and L2 regularizations to prevent overfitting and improve generalization

- **Transfer Learning:**
  - Provides the ability to reuse pre-trained models on new tasks, allowing for faster and more efficient model training

- **Evaluation:**
  - Provides comprehensive statistics, including confusion matrix, to evaluate the performance of the trained model


- **Extensible Framework:**
  - Users can extend the functionality of the neural network by adding their own activation functions, cost functions, and optimizers using function pointers

- **Automake Build System:**
  - Uses Automake as the build system, which streamlines the development process and supports easy project scalability for expanding feature sets


## Building the Project

### Prerequisites

To build and run NeuralNet, ensure that you have the following dependencies installed:

- C compiler (gcc)
- GNU Make
- Automake

### Installation

1. Clone the NeuralNet repository from GitHub:

   ```shell
   git clone https://github.com/matcor852/NeuralNet
   ```

2. Navigate to the project directory:

   ```shell
   cd NeuralNet
   ```

3. Build the project:

   ```shell
   autoreconf --install
   ./configure
   make
   ```

### Getting Started

To create and train a neural network using this implementation, refer to the  XOR example in the `src/main.c` source file to understand how to customize the network for your specific use case.

### Contributing

If you encounter any bugs or have suggestions for improvements, please report them to `matcor852@gmail.com`. However, please note that this project is primarily a learning exercise and should not be used in any production environment without significant enhancements.

## License

The NeuralNet project is released under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

## Disclaimer

While efforts have been made to ensure the correctness and efficiency of the network, it should be noted that this work is primarily intended for educational purposes.

The NeuralNet project may contain bugs, limitations, or performance issues that make it unsuitable for use in production environments. It is strongly recommended to conduct thorough testing and make necessary improvements before considering its deployment in any critical or commercial systems.

The developer of NeuralNet do not assume any responsibility for any damages or losses incurred from the use or misuse of this software. Users are solely responsible for evaluating its suitability for their specific needs and are advised to exercise caution when utilizing it in any context.

