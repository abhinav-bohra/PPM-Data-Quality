import os
import glob
import warnings
import argparse
import torch, pickle
import pandas as pd
warnings.filterwarnings('ignore')
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
torch.set_printoptions(profile="full")
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

task_names = {
  'preds-next_step_prediction.pickle': 'NEXT ACTIVITY',
  'preds-outcome_prediction.pickle': 'OUTCOME ACTIVITY',
  'preds-next_resource_prediction.pickle': 'NEXT RESOURCE',
  'preds-last_resource_prediction.pickle': 'LAST RESOURCE',
  'preds-duration_to_next_event_prediction.pickle': 'DURATION TO NEXT EVENT',
  'preds-duration_to_end_prediction.pickle': 'DURATION TO END'
}

#--------------------------------------------------------------------------------------------
# Get Ground Truth and Predictions
# data[0] -> Inputs,  data[1] -> Preds,  data[2] -> Targets (True Outputs)
#--------------------------------------------------------------------------------------------

path = f"{folder}/models/run0"
logs = os.listdir(path)
results = {}

for log in logs:
  models = os.listdir(f"{path}/{log}")
  for model in models: 
    print(model, log)
    case_results = {
      'log': list(),
      'model': list(),
      'case_len': list(),
      'NEXT ACTIVITY ACC': list(),'NEXT ACTIVITY PRE': list(),'NEXT ACTIVITY REC': list(),'NEXT ACTIVITY F1': list(),
      'OUTCOME ACTIVITY ACC': list(),'OUTCOME ACTIVITY PRE': list(),'OUTCOME ACTIVITY REC': list(),'OUTCOME ACTIVITY F1': list(),
      'NEXT RESOURCE ACC': list(),'NEXT RESOURCE PRE': list(),'NEXT RESOURCE REC': list(),'NEXT RESOURCE F1': list(),
      'LAST RESOURCE ACC': list(),'LAST RESOURCE PRE': list(),'LAST RESOURCE REC': list(),'LAST RESOURCE F1': list(),
      'DURATION TO NEXT EVENT MSE': list(),
      'DURATION TO END MSE': list()
    }

    #Get case_len
    pred_file = pd.read_pickle(f"{path}/{log}/{model}/preds-next_step_prediction.pickle") #Consider only categorical columns
    xyz = [int(torch.count_nonzero(row)) for row in pred_file[0][-1]]
    print(xyz)
    print(pred_file[0][-1])
    num_cases = len(set([int(torch.count_nonzero(row)) for row in pred_file[0][-1]]))
    case_results['log'] = [log]*num_cases
    case_results['model'] = [model]*num_cases
    result_na = ['NA']*num_cases
    for task in task_names:
      preds_path = f"{path}/{log}/{model}/{task}"
      if os.path.exists(preds_path):        
        data = pd.read_pickle(preds_path)
        if "duration" in task:
          preds = torch.squeeze(data[1]).tolist()
        else:
          preds =[torch.argmax(data[1][i],dim=0).item() for i in range(0,data[1].size(0))]
        targs = data[2].tolist()
        case_len =[int(torch.count_nonzero(row)) for row in data[0][-1]]
        assert len(preds) == len(targs)
        assert len(preds) == len(case_len)

        columns = ['case_len', f'pred_{task_names[task]}',f'targ_{task_names[task]}']
        data = [case_len, preds, targs]        
        df_dict = dict()
        for d,c in zip(data, columns):
          df_dict[c] = d
        try:
          df = pd.DataFrame(df_dict)
        except Exception as e:
          print(e)
          print(model,log,task)
          print(len(preds))
          print(len(targs))
          print(len(case_len))
          print(len(data))
      
        #--------------------------------------------------------------------------------------------
        # Group by case_len and evaluate
        #--------------------------------------------------------------------------------------------
        groups = df.groupby('case_len')
        case_len_grps = list(groups.groups.keys())
        case_results['case_len'] = case_len_grps
        for case in case_len_grps:
          case_len_df = groups.get_group(case)
          if "duration" not in task:
            cr = classification_report(case_len_df[ f'targ_{task_names[task]}'], case_len_df[ f'pred_{task_names[task]}'], output_dict = True)
            case_results[f'{task_names[task]} ACC'].append(cr['accuracy'])
            case_results[f'{task_names[task]} PRE'].append(cr['macro avg']['precision'])
            case_results[f'{task_names[task]} REC'].append(cr['macro avg']['recall'])
            case_results[f'{task_names[task]} F1'].append(cr['macro avg']['f1-score'])
          else:
            y_true = case_len_df[f'targ_{task_names[task]}']
            y_pred = case_len_df[f'targ_{task_names[task]}']
            mse = mean_squared_error(y_true,y_pred)
            case_results[f'{task_names[task]} MSE'].append(mse)
      else:
        if "duration" not in task:
          case_results[f'{task_names[task]} ACC']=result_na
          case_results[f'{task_names[task]} PRE']=result_na
          case_results[f'{task_names[task]} REC']=result_na
          case_results[f'{task_names[task]} F1']=result_na
        else:
          case_results[f'{task_names[task]} MSE']=result_na

    results[f'{log}_{model}'] = pd.DataFrame(case_results)

with pd.ExcelWriter(f'{folder}/case_eval.xlsx') as writer:
  for key in results:
     results[key].to_excel(writer, sheet_name=key)
     
print("[SUCCESS] Computed Case level Results successfully")