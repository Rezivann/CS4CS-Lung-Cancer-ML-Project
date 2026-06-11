# CS4CS Capstone Project: ML Models Evaluating Lung Cancer Severity

This project compares three different types of machine learning models for lung cancer severity classification using tabular medical data from [Demeke Endalie and Wondmagegn Taye Abebe's study](https://journals.plos.org/digitalhealth/article?id=10.1371/journal.pdig.0000308).

## Features
- Implements Naïve Bayes, feedforward neural network, and TabNet classifiers using Scikit-learn, TensorFlow, and PyTorch
- Predicts lung cancer severity given tabular medical data


## Methodology
Each model was trained on 80% of the data and evaluated on the other 20%. For ten trials, this process was repeated with shuffled data, and final accuracy scores indicate average performance across all 10 trials.

## Average Accuracies Across 10 Trials
1. TabNet Classifier: 98.70%
2. Feedforward Neural Network: 96.55%
3. Naïve Bayes: 90.55%


## Getting Started

### Prerequisites
- Python 3.11 or higher
- pip package manager

### Installation
1. **Clone the repository**
```bash
git clone https://github.com/Rezivann/CS4CS-Lung-Cancer-ML-Project
cd CS4CS-Lung-Cancer-ML-Project
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```
## Usage

1. **Run the notebooks**

Open the notebook files in `Scripts/PythonNotebookScripts` and run all of the cells in each one.

## Outputs

Each notebook trains and tests its corresponding model 10 times and prints the accuracy after each trial. Then, the accuracies of the 10 trials are averaged to produce a final, consistent accuracy score.

## Conclusions

The TabNet classifier performed the best out of all three models in terms of average accuracy. The feedforward neural network also performed well, while the Naïve Bayes classifier attained the lowest average accuracy.

However, the TabNet and Naïve Bayes classifiers may prove especially helpful in the medical field because of their feature importance mechanisms. These mechanisms indicate which features were most impactful to classification, making these models more understandable when used in medical contexts.



