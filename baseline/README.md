Baselines for Comparison

We utilized two baselines for performance and accuracy comparison:
	1.	LeNet-from-scratch: A Python implementation of the LeNet model built entirely from scratch without using external libraries. (# Taken code from Git - https://github.com/mattwang44/LeNet-from-Scratch.git for Baseline)
	2.	Tensor_baseline: An implementation of LeNet using PyTorch tensors.

Our program exceeds the performance of LeNet-from-Scratch, completing one epoch in approximately 20 seconds on both GPU and CPU while LeNet-from-Scratch takes 180 secs for one epoch and matches the accuracy of both baseline models.
