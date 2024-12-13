# GAS: Generative Activation-Aided Asynchronous Split Federated Learning

This repository is the demo of GAS.

## Requirements

To install the required packages:

```setup
pip install -r requirements.txt
```

## Training and Evaluation

To train and evaluate the model(s) in the paper, run this command:

```train
python GAS_main.py 
```

## Results

Our model achieves the following performance on CIFAR-10, CIFAR100, CINIC10 and Fashion-MNIST:

| Dataset       | $s=2$        | $\alpha=0.1$ |
| ------------- | ------------ | ------------ |
| CIFAR10       | $82.78±0.58$ | $81.72±0.50$ |
| CINIC10       | $68.32±0.17$ | $65.94±1.14$ |
| Fashion-MNIST | $90.66±0.20$ | $90.58±0.34$ |

