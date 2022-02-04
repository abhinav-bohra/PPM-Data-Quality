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

### Class Imbalance

NOTE: Run all of the following commands from IBM-Data-Quality folder

Step 1: Basic Set-up
Set models and datasets in 'default' (line 58: prediction_evaluation.py)
Set the same models and datasets in 'CI' mode as well.(line 51: prediction_evaluation.py)
List of supported models and datasets - [link]list (Please use the exact names)

```
export CUDA_VISIBLE_DEVICES=0,1    #Specify GPU number(s)
```

Step 2: Get Default Results

```
!python prediction_evaluation.py --exp default --save_folder results_default
```

Step 3: Compute Class Imbalance Score on default features

```
! python class_imbalance.py --folder results_default 
```

Step 4: Compute Case-level results

```
! python case_eval.py --folder results_default 
```

Step 5: Get results after class imbalance remiditions (undersampling)

```
!python prediction_evaluation.py --exp CI --balancing_technique NM --save_folder results_nm
!python prediction_evaluation.py --exp CI --balancing_technique CONN --save_folder results_conn
!python prediction_evaluation.py --exp CI --balancing_technique NCR --save_folder results_ncr
```

Step 6: Compute Class Imbalance Score on undersampled features

```
!python class_imbalance.py --folder results_nm
!python class_imbalance.py --folder results_conn
!python class_imbalance.py --folder results_ncr
```

Step 7: Compute Case-level results

```
!python case_eval.py --folder results_nm
!python case_eval.py --folder results_conn
!python case_eval.py --folder results_ncr
```


### Class Overlap
Step 1: Compute Class Overlap (F1 & F2) Score on features

```
!python class_overlap.py --folder results_defaults
```



For each model, checkpoints are saved as .pth files in 01_Missing-Values/results/models/

(It saves the model's best during training)

<hr>

The repository is primarily built upon the MPPN Repository -> https://github.com/joLahann/mppn