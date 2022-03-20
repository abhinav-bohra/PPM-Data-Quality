# PPM-Data-Quality

> This repository contains the source code of the paper Assessing Quality of Event Logs for Business Process Predictions.


## Installation

Clone the repo and run the following commands

```
cd PPM-Data-Quality
conda create --name ppmdq python=3.7.
conda activate ppmdq
python setup.py install
pip install -r requirements.txt
```

## Experiments

Run the following commands from PPM-Data-Quality folder

```
export CUDA_VISIBLE_DEVICES=0,1  #Specify GPU number(s)
```

### I. Class Imbalance

Step 1: Basic Set-up

Set models and logs in 'default' mode (line 58: prediction_evaluation.py)  
Set the same models and logs in 'CI' mode as well (line 51: prediction_evaluation.py)

Step 2: Get Default results

```
python prediction_evaluation.py --exp default --save_folder results_default
```

Step 3: Compute Class Imbalance Score on default features

```
python class_imbalance_target.py --folder results_default 
```

Step 4: Compute Case-level results

```
python case_eval.py --folder results_default 
```

Step 5: Get results after class imbalance remediations (undersampling)

```
python prediction_evaluation.py --exp CI --balancing_technique NM --save_folder results_nm
python prediction_evaluation.py --exp CI --balancing_technique CONN --save_folder results_conn
python prediction_evaluation.py --exp CI --balancing_technique NCR --save_folder results_ncr
```

Step 6: Compute Class Imbalance Score on undersampled features

```
python class_imbalance_target.py --folder results_nm
python class_imbalance_target.py --folder results_conn
python class_imbalance_target.py --folder results_ncr
```

Step 7: Compute Case-level results after undersampling

```
python case_eval.py --folder results_nm
python case_eval.py --folder results_conn
python case_eval.py --folder results_ncr
```


### II. Class Overlap
Step 1: Compute Class Overlap (F1 & F2) Score on default features

```
python class_overlap.py --folder results_default
```


### III. Missing Values

Step 1:  Set models and pre-processed logs in 'MV' mode (line 58: prediction_evaluation.py)  
Step 2:  Get results on logs with filled missing values

```
python prediction_evaluation.py --exp MV --save_folder results_missing_values
```


### IV. Outlier Filtering


Step 1: To only filter outliers
```
python prediction_evaluation.py --exp default --filter_percentage 10 --save_folder results_outliers
```

Step 2: To filter outliers first and then balance the dataset

```
python prediction_evaluation.py --exp CI --filter_percentage 10 --balancing_technique NM --save_folder results
```


<br>
<br>
NOTE: For each model, checkpoints are saved as .pth files in results_folder/models/run0/

(It saves the model's best during training)

<hr>

The repository is primarily built upon the MPPN Repository -> https://github.com/joLahann/mppn
