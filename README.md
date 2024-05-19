# Edge Server Deployment for Hierarchical Federated Learning with Non-IID Data Distributions

## Overview

This repository contains the code and resources for the paper titled "Edge Server Deployment for Hierarchical Federated Learning with Non-IID Data Distributions," accepted at the Joint Conference on Communications and Information (JCCI) 2024. This project aims to enhance the performance of Hierarchical Federated Learning (HFL) by modifying the edge server deployment method in non-independent and identically distributed (Non-IID) settings.

## Project Description

In this project, we focus on improving HFL by strategically deploying edge servers to handle Non-IID data distributions. We use the MNIST dataset as our test dataset, simulating clients within a large M x N pixel grid. 

### Key Concepts

- **Non-IID Data Distribution**: We simulate Non-IID data by assigning labels to specific pixels based on a multivariate normal distribution. Each axis (M and N) independently follows a Gaussian Mixture Model, creating joint probabilities for label presence in the dataset.
- **Edge Server Deployment**: We explore different configurations by varying the number of edge servers (1, 4, 8, 16) and analyze their impact on the learning process.
- **Federated Averaging Algorithm**: We employ the federated averaging algorithm to aggregate the model updates from the clients.
- **Performance Measurement**: We measure performance by the number of local epochs required to reach a target accuracy of 0.965. We then calculate the time delay using the Free Space Path Loss mathematical expression to determine the optimal deployment strategy.

## Implementation Details

### Dataset

- **MNIST Dataset**: Used as the test dataset for this project.

### Client Setup

- **Client Grid**: Clients are set up in a large M x N pixel grid.
- **Client Selection**: 100 clients are randomly selected from the grid.
- **Label Probabilities**: Labels are assigned based on joint probabilities created by the independent Gaussian Mixture Models on the M and N axes.

### Edge Server Deployment

- **Configurations**: We experiment with 1, 4, 8, and 16 edge servers to determine the best deployment strategy.

### Federated Learning

- **Algorithm**: Federated Averaging is used to aggregate client model updates.
- **Target Accuracy**: The goal is to reach an accuracy of 0.965.
- **Local Epochs**: The number of local epochs required to reach the target accuracy is recorded.

### Performance Evaluation

- **Time Delay Calculation**: Time delay is calculated using the Free Space Path Loss mathematical expression.
- **Comparison**: Total delays for different configurations are compared to identify the most efficient edge server deployment strategy.

## Getting Started

### Prerequisites

- Python 3.x
- Required Python libraries (listed in `requirements.txt`)

### Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/hierarchical-federated-learning.git
   cd hierarchical-federated-learning
2. Install the required Library


### Running the Code
1. Prepare the dataset:

   Download and preprocess the MNIST dataset.

   
2. Configure the parameters:

   Modify the configuration file (config.json) to set the number of edge servers, client grid size, and other parameters.

3.Execute the main script:

python main.py --config config.json

### Results

The results will include the number of local epochs required to reach the target accuracy for each edge server configuration and the calculated time delays.

## Citation
If you use this code or dataset in your research, please cite the following paper:

@inproceedings{JCCI_2024,
  title={Edge Server Deployment for Hierarchical Federated Learning with Non-IID Data Distributions},
  author={SunWoo Kang,Minseok Choi},
  booktitle={Joint Conference on Communications and Information (JCCI)},
  year={2024}
}
Contact
For any questions or inquiries, please contact:

Your Name: swkang.official@gamil.com

Thank you for your interest in this project!

