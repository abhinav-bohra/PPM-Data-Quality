import os
import warnings
import argparse
import torch, pickle
import pandas as pd
warnings.filterwarnings('ignore')
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error

#------------------------------------------------------------------------------------------
# Command Line Arguments
#------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--folder', default="results_default",type=str, help='results folder')    

#------------------------------------------------------------------------------------------
# Arguments and Global Variables
#------------------------------------------------------------------------------------------
args = parser.parse_args()
folder = args.folder

column_names = {
  'preds-next_step_prediction.pickle': 'NEXT ACTIVITY',
  'preds-outcome_prediction.pickle': 'OUTCOME ACTIVITY',
  'preds-next_resource_prediction.pickle': 'NEXT RESOURCE',
  'preds-last_resource_prediction.pickle': 'LAST RESOURCE',
  'preds-duration_to_next_event_prediction.pickle': 'DURATION TO NEXT EVENT',
  'preds-duration_to_end_prediction.pickle': 'DURATION TO END'
}

#--------------------------------------------------------------------------------------------
# Get Ground Truth and Predictions
# dls[0] -> Inputs,  dls[1] -> Preds,  dls[2] -> Targets (True Outputs)
#--------------------------------------------------------------------------------------------
results = list()
path = f"{folder}/models/run0"
logs = os.listdir(path)
for log in logs:
  models = os.listdir(f"{path}/{log}")
  for model in models: 
    targets = glob.glob(f"{path}/{log}/{model}/preds-*.pickle")
    files = [t.split('\\')[1] for t in targets if "setup" not in t]
    target_dfs = list()
    for file in files:
      dataset_path = f"{path}/{log}/{model}/{file}"
      
    dls = pd.read_pickle(rf'{path}/{log}/{model}/preds.pickle')
    PPObj = pd.read_pickle(rf'{path}/{log}/{model}/PPObj.pickle')
    x_input = dls[0] 
    case_len =[int(torch.count_nonzero(ele[0])) for ele in x_input[0]]
    num_cats = len(PPObj.ycat_names)
    num_conts = len(PPObj.ycont_names)
    preds =[torch.argmax(dls[1][i],dim=1) for i in range(0,num_cats)]
    preds.append(torch.squeeze(dls[1][num_cats]))
    preds = tuple(preds)
    targs = dls[2]
    preds = ([pred.tolist() for pred in preds])
    targs = ([targ.tolist() for targ in targs])

    y_cols = list()
    for y in PPObj.y_names:
      if y in PPObj.ycat_names:
        y_cols.append("CAT_" + y)
      if y in PPObj.ycont_names:
        y_cols.append("CONT_" + y)

    columns = ['case_len'] + [f"pred_{y}" for y in y_cols] + [f"targ_{y}" for y in y_cols]
    data = [case_len] + preds + targs
    df_dict = dict()
    for d,c in zip(data, columns):
      df_dict[c] = d
    df = pd.DataFrame(df_dict)
    
    #--------------------------------------------------------------------------------------------
    # Group by case_len and evaluate
    #--------------------------------------------------------------------------------------------
    groups = df.groupby('case_len')
    case_len_grps = list(groups.groups.keys())
    for case in case_len_grps:
      case_len_df = groups.get_group(case)
      result = [log, model, case]
      for y in PPObj.ycat_names:
        cr = classification_report(case_len_df[f'targ_CAT_{y}'], case_len_df[f'pred_CAT_{y}'], output_dict = True)
        result = result + ([cr['accuracy'], cr['macro avg']['precision'], cr['macro avg']['recall'], cr['macro avg']['f1-score']])
      
      for y in PPObj.ycont_names:
        result = result + ([mean_squared_error(case_len_df[f'targ_CONT_{y}'], case_len_df[f'pred_CONT_{y}'])])
      
      results.append(result)

# Columns
columns = ['Dataset','Model','Case Length']
measures = ['Acc', 'Pre', 'Rec', 'F1']
for y in PPObj.ycat_names:
	for m in measures:
		columns.append(f'{y}_{m}')

for y in PPObj.ycont_names:
	columns.append(f'{y}_MSE')

df_results = pd.DataFrame(results, columns = columns)
df_results.to_csv(f"{folder}/case_eval.csv",index=False)
print("[SUCCESS] Computed Case level Results successfully")