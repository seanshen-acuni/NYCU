# Gradient Checking: NumPy vs. PyTorch

## Description
This project implements a simple feedforward neural network **from scratch in NumPy** and compares its backpropagation gradients with **PyTorch’s autograd** results.  
It demonstrates how to build modular layers, implement forward/backward propagation, and validate gradient correctness.  

The project is intended as an **educational resource** for learning about backpropagation and gradient checking.

## Architecture
- Input Layer: 784 neurons (28×28 MNIST-like)  
- Hidden Layers: 2 × 128 neurons with ReLU activation  
- Output Layer: 10 neurons with Softmax activation  
- Loss Function: Cross-Entropy Loss

## Output
Example output during training:
```
iter: 00, loss: 2.32081
iter: 01, loss: 2.30756
...
rtol: 1.23E-07
rtol: 8.45E-08
Final mean rtol ≈ 1e-07
```

## Files

## Installation
Clone the repository and install dependencies:
  ```
  pip install -r requirements.txt
  ```

## Requirements
- Python 3.10+
- NumPy
- PyTorch
- tqdm

## Usage
1. Run the NumPy implementation:
  ```
  python test_np.py --n 100 --output_dir ./grads_np
  ```
2. Run the PyTorch implementation:
  ```
  python test_pt.py --n 100 --output_dir ./grads_pt
  ```
3. Compare gradients:
  ```
  python diff.py --grads_np ./grads_np --grads_pt ./grads_pt --verbose
  ```

## Roadmap
- Add more activation functions (Tanh, LeakyReLU)
- Extend to Convolutional Neural Networks (CNN)
- Implement numerical gradient checking
- Add unit tests

## Contributing
Contributions are welcome!

1. Fork this repository
2. Create a new branch: git checkout -b feature/my-feature
3. Commit your changes
4. Open a Pull Request

## Authors and Acknowledgment
Author: Sean Shen (沈昱翔)

## License
This project is licensed under the MIT License.

## Project Status
This project was developed as part of my learning and is shared for demonstration purposes. No active maintenance is planned.
