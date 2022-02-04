import warnings
import argparse
import torch, pickle
import pandas as pd
warnings.filterwarnings('ignore')
from sklearn.metrics import classification_report

#------------------------------------------------------------------------------------------
# Command Line Arguments
#------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="BPIC_12",type=str, help='name of dataset')    
parser.add_argument('--model', default="MPPN",type=str, help='name of model')        
parser.add_argument('--mode', default="0",type=str, help='experiment mode')   

#------------------------------------------------------------------------------------------
# Arguments and Global Variables
#------------------------------------------------------------------------------------------
args = parser.parse_args()
mode = args.mode
dataset = args.dataset 
model = args.model

#--------------------------------------------------------------------------------------------
# Get Ground Truth and Predictions
# obj[0] -> Inputs,  obj[1] -> Preds,  obj[2] -> Targets (True Outputs)
#--------------------------------------------------------------------------------------------

obj = pd.read_pickle(rf'test/models/run0/{dataset}/{model}/preds.pickle')
x_input = obj[0] 
case_len =[int(torch.count_nonzero(ele[0])) for ele in x_input[0]]
num_cats = x_input[0].size(1)
num_conts = x_input[1].size(1)

preds =[torch.argmax(obj[1][i],dim=1) for i in range(0,num_cats)]
preds.append(torch.squeeze(obj[1][num_cats]))
preds = tuple(preds)
targs = obj[2]
preds = tuple([pred.tolist() for pred in preds])
targs = tuple([targ.tolist() for targ in targs])

fields = ['Case Length'] + [f"pred_{t}" for t in range(1, target_cols] + [f"{t}_F2" for t in target_cols]

data = {
    'case_len': case_len,
    'pred_act': preds[0],
    'pred_res': preds[1],
    'pred_time': preds[2],
    'targ_act': targs[0],
    'targ_res': targs[1],
    'targ_time': targs[2],
}
df = pd.DataFrame(data)

#--------------------------------------------------------------------------------------------
# Group by case_len and evaluate
#--------------------------------------------------------------------------------------------
groups = df.groupby('case_len')
case_len_grps = list(groups.groups.keys())
results = list()
for case in case_len_grps:
  case_len_df = groups.get_group(case)
  cr_act = classification_report(case_len_df['targ_act'], case_len_df['pred_act'], output_dict = True)
  cr_res = classification_report(case_len_df['targ_res'], case_len_df['pred_res'], output_dict = True)
  results.append([case, cr_act['accuracy'], cr_act['macro avg']['precision'], cr_act['macro avg']['recall'], cr_act['macro avg']['f1-score'], cr_res['accuracy'], cr_res['macro avg']['precision'], cr_res['macro avg']['recall'], cr_res['macro avg']['f1-score']])
  
df_results = pd.DataFrame(results, columns = ['Case Len','Activity Acc', 'Activity Pre', 'Activity Rec', 'Activity F1', 'Resource Acc', 'Resource Pre', 'Resource Rec', 'Resource F1'])
df_results.to_csv(f"test/models/run0/{dataset}/{model}/case_eval.csv",index=False)

#------------------------------------------------------------------------------------------
# Save Results
#------------------------------------------------------------------------------------------
with open(f"CaseResults-{dataset}-{model}-{mode}.csv", 'w', newline='') as csvfile: 
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields) 
    csvwriter.writerows(rows)

print("[SUCCESS] Computed Case level Results successfully")