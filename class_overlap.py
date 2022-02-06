import glob
import os, warnings
import csv, argparse
import pandas as pd
import numpy as np
from class_overlap_complexity import MFEComplexity
warnings.filterwarnings("ignore")

##------------------------------------------------------------------------------------------
# Command Line Arguments
#------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--folder', default="results_default",type=str, help='results folder')    

#------------------------------------------------------------------------------------------
# Arguments and Global Variables
#------------------------------------------------------------------------------------------
args = parser.parse_args()
folder = args.folder
path = f"{folder}/models/run0"
logs = os.listdir(path)

column_names = {
  'targets-next_step_prediction.csv': 'NEXT ACTIVITY',
  'targets-outcome_prediction.csv': 'OUTCOME ACTIVITY',
  'targets-next_resource_prediction.csv': 'NEXT RESOURCE',
  'targets-last_resource_prediction.csv': 'LAST RESOURCE',
  'targets-duration_to_next_event_prediction.csv': 'DURATION TO NEXT EVENT',
  'targets-duration_to_end_prediction.csv': 'DURATION TO END'
}

rows = list()
for log in logs:
	models = os.listdir(f"{path}/{log}")
	for model in models:
		#------------------------------------------------------------------------------------------
		# Features and Targets
		#------------------------------------------------------------------------------------------
		#Features
		df = pd.read_csv(f"{path}/{log}/{model}/features.csv")
		df = df.loc[:, (df != 0).any(axis=0)] #Drop columns with no non-zero value
		df = df.dropna() #Drop null values, if any
		try:
			features = df.drop(columns=["case_len"]) #Drop Case_len column as it is not a feature
		except:
			print(log,model)
			continue
		feature_cols = list(features.columns) # feature names 
		#Targets
		targets = glob.glob(f"{path}/{log}/{model}/targets*.csv")
		files = [t.split('\\')[1] for t in targets if "setup" not in t] #Consider only categorical columns
		target_dfs = list()
		for file in files:
			dataset_path = f"{path}/{log}/{model}/{file}"
			df_ = pd.read_csv(dataset_path)
			df_ = df_.dropna()
			df_ = df_.rename(columns={df_.columns[0]:column_names[file]})
			target_dfs.append(df_)
      
		targets = pd.concat(target_dfs, axis=1)
		target_cols = targets.columns.tolist()
		#Checking if both are aligned or not
		assert len(features[feature_cols[0]]) == len(targets[target_cols[0]])

		#------------------------------------------------------------------------------------------
		# Computing F1 & F2 Score
		#------------------------------------------------------------------------------------------
		X = features.to_numpy()
		f1_scores = list()
		f2_scores = list()

		for target in target_cols:
			y = targets[target].to_numpy()
			mfe = MFEComplexity()
			try:
				f1_all, f1_max, f1_nanmax = mfe.ft_f1(X,y)
			except Exception as e:
				f1_all, f1_max, f1_nanmax = None,None,None
				print("[ERROR][Class Overlap F1] -", e)
			try:
				f2 = mfe.ft_f2(X,y)
				f2 = np.nanmean(f2)
			except Exception as e:
				f2 = None
				print("[ERROR][Class Overlap F2] -", e)

			f1_scores.append(f1_nanmax)
			f2_scores.append(f2)

		rows.append([log, model, "DATASET-LEVEL"] + f1_scores + f2_scores)
		print(f"[SUCCESS][{log}][{model}] Computed Dataset level Class overlap successfully")

		#------------------------------------------------------------------------------------------
		# Case Level Overlap
		#------------------------------------------------------------------------------------------
		df_full = pd.concat([df, targets], axis=1)
		groups = df_full.groupby('case_len')
		case_len_grps = list(groups.groups.keys())
		for case in case_len_grps:
			case_len_df = groups.get_group(case)
			case_level_targets  = case_len_df[target_cols]
			try:
				case_level_features = case_len_df.drop(columns=["case_len"] + target_cols) #Drop Case_len & target column as it is not a feature
			except:
				print(case_len_df.head(1))
			X = case_level_features.to_numpy()
			f1_scores = list()
			f2_scores = list()
			for target in target_cols:
				y = case_level_targets[target].to_numpy()
				mfe = MFEComplexity()
				try:
					f1_all, f1_max, f1_nanmax = mfe.ft_f1(X,y)
				except Exception as e:
					f1_all, f1_max, f1_nanmax = None,None,None
					print("[ERROR][Class Overlap F1] -", e)
				try:
					f2 = mfe.ft_f2(X,y)
					f2 = np.nanmean(f2)
				except Exception as e:
					f2 = None
					print("[ERROR][Class Overlap F2] -", e)

				f1_scores.append(f1_nanmax)
				f2_scores.append(f2)

			rows.append([log, model, case] + f1_scores + f2_scores)
		print(f"[SUCCESS][{log}][{model}] Computed Case level Class overlap successfully")

fields = ['Dataset', 'Model', 'Case Length'] + [f"{t}_F1" for t in target_cols] + [f"{t}_F2" for t in target_cols]
#------------------------------------------------------------------------------------------
# Save Results
#------------------------------------------------------------------------------------------
with open(f"{folder}/class_overlap_results.csv", 'w', newline='') as csvfile: 
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields) 
    csvwriter.writerows(rows)
