# IBM-Data-Quality

> This repository contains the source code of the paper Assessing Quality of Event Logs for Business Process Predictions.


## Installation

Clone the repo and run the following commands
```
cd IBM-Data-Quality
conda create --name idq python=3.7.
conda activate idq
python setup.py install
pip install -r requirements.txt
```

## Experiments

```
python prediction_evaluation.py --exp MV --save_folder results
```

--exp : Experiment Mode
- MV for Missing Values
- CI for Class Imbalance
- test for Testing

--save_folder : Name of folder to save model checkpoints and results.
Model Checkpoints are saved as .pth files in 01-Missing-Values/resutls/models/ for every model.

The repository is primarily built upon the MPPN Repository -> https://github.com/joLahann/mppn
