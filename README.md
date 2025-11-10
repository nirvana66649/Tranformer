# Transformer

This repository contains implementations of Transformer-based models using two different deep learning frameworks: MindSpore and PyTorch. These implementations are inspired by the "Attention Is All You Need" paper (NIPS 2017) and demonstrate the training and evaluation of Transformer models for language modeling tasks.

## Project Structure

```
├── Mindspore_Transformer_TrainNInterface.ipynb  # Transformer implementation using MindSpore
├── transformer_llm（pytorch）.ipynb             # Transformer implementation using PyTorch
├── transformer_model.ckpt                       # Checkpoint file for the trained MindSpore model
├── README.md                                    # Project documentation
├── tansformer_part(mindspore).ipynb             # Additional MindSpore implementation (optional)
├── NIPS-2017-attention-is-all-you-need-Paper.pdf # Reference paper
```

## Notebooks Overview

### 1. MindSpore Implementation
**File:** `Mindspore_Transformer_TrainNInterface.ipynb`

This notebook demonstrates the implementation of a Transformer-based language model using the MindSpore framework. Key features include:

- **Model Architecture:**
  - Multi-head self-attention mechanism.
  - Feed-forward neural networks.
  - Layer normalization and residual connections.
  - Positional encoding.

- **Training Details:**
  - Dataset: A text dataset is loaded and tokenized using the `tiktoken` library.
  - Hyperparameters: Adjustable parameters such as batch size, learning rate, number of layers, and dropout.
  - Optimizer: Adam with weight decay.
  - Training loop with periodic evaluation on validation data.

- **Output:**
  - The model is trained for 1000 steps, and the final checkpoint is saved as `transformer_model.ckpt`.

### 2. PyTorch Implementation
**File:** `transformer_llm（pytorch）.ipynb`

This notebook provides an alternative implementation of a Transformer-based language model using the PyTorch framework. Key features include:

- **Model Architecture:**
  - Similar to the MindSpore implementation, with multi-head self-attention, feed-forward layers, and positional encoding.

- **Training Details:**
  - Dataset preparation and tokenization.
  - Training loop with loss computation and backpropagation.

- **Output:**
  - The notebook demonstrates the training process and evaluation of the model.

## Requirements

To run the notebooks, ensure you have the following dependencies installed:

- Python 3.x
- MindSpore
- PyTorch
- NumPy
- tiktoken
- Requests

You can install the required Python packages using pip:

```bash
pip install mindspore pytorch numpy tiktoken requests
```

## Usage

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Tranformer
   ```

2. Open the desired notebook in Jupyter Notebook or JupyterLab:
   ```bash
   jupyter notebook Mindspore_Transformer_TrainNInterface.ipynb
   ```

3. Follow the instructions in the notebook to train and evaluate the Transformer model.

## Reference

- [Attention Is All You Need (NIPS 2017)](https://arxiv.org/abs/1706.03762)

## Notes

- The MindSpore implementation includes additional warnings about deprecated parameters (e.g., `keep_prob`), which should be updated to the latest API (`p`).
- The PyTorch implementation is designed to run on GPUs for faster training.

Feel free to explore and modify the notebooks to suit your specific use case.
