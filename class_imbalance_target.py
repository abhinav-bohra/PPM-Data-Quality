import glob
import os,csv
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from class_imbalance_degree import imbalance_degree
#------------------------------------------------------------------------------------------
# Command Line Arguments
#------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--folder', default="results_default",type=str, help='results folder')    

#--------------------------------------------------------------------------------------------
# Helper Function
#--------------------------------------------------------------------------------------------
def getMajorityDist(values, num_classes, size):
  avg = size/num_classes
  count = 0
  for val in values:
    if val >= avg:
      count=count+1

  if count >= num_classes/2:
    # return f"Multi-majority with {count} major classes and {num_classes-count} minor classes."
    return "Multi-majority", round((num_classes-count)/num_classes,2)*100
  else:
    # return f"Multi-minority with {count} major classes and {num_classes-count} minor classes."
    return "Multi-minority", round((num_classes-count)/num_classes,2)*100

#--------------------------------------------------------------------------------------------
# Class Imbalance Analysis Function
#--------------------------------------------------------------------------------------------
def class_imbalance_analysis(df,cat_col):
  size = len(df)
  num_classes = len(df[cat_col].unique())
  frequency_dist = df[cat_col].value_counts().values
  empirical_dist= frequency_dist/size
  ir = round(max(frequency_dist)/min(frequency_dist)) if min(frequency_dist)!=0 else "INF"
  multi_group, minority_percent = getMajorityDist(frequency_dist, num_classes, size)
  id_scores = list()
  sim_func = ["EU","CH","KL","HE","TV","CS"]
  for sf in sim_func:
    id_scores.append(imbalance_degree(np.array(df[cat_col]), sf))
  metrics = [size,num_classes,ir, multi_group, minority_percent]
  metrics = metrics + id_scores
  return metrics

#--------------------------------------------------------------------------------------------
# Main Function
#--------------------------------------------------------------------------------------------
column_names = {
  'targets-next_step_prediction.csv': 'NEXT ACTIVITY',
  'targets-outcome_prediction.csv': 'OUTCOME ACTIVITY',
  'targets-next_resource_prediction.csv': 'NEXT RESOURCE',
  'targets-last_resource_prediction.csv': 'LAST RESOURCE'
}

if __name__ == "__main__":
  fields = ["Dataset","Model","Column","Size of Dataset","Num Classes","Imbalance Ratio",\
   "Multi-Minority/Majority","Minority Class", "Euclidean distance", "Chebyshev distance",\
 "Kullback Leibler divergence","Hellinger distance","Total variation distance","Chi-square divergence"]
  rows = list()
  args = parser.parse_args()
  folder = args.folder
  path = f"{folder}/models/run0"
  logs = os.listdir(path)
  for log in logs:
    models = os.listdir(f"{path}/{log}")
    for model in models:
      #Concatante all target dataframes
      targets = glob.glob(f"{path}/{log}/{model}/targets*.csv")
      files = [t.split('\\')[1] for t in targets if "duration" not in t and "setup" not in t] #Consider only categorical columns
      target_dfs = list()
      for file in files:
        dataset_path = f"{path}/{log}/{model}/{file}"
        df = pd.read_csv(dataset_path)
        df = df.dropna()
        df = df.rename(columns={df.columns[0]:column_names[file]})
        target_dfs.append(df)
      
      df_final = pd.concat(target_dfs, axis=1)
      all_cols = df_final.columns.tolist()
      for col in all_cols:
        row = [log,model,col] + (class_imbalance_analysis(df_final,col))
        rows.append(row)

  with open(f"{folder}/class_imbalance_results.csv",'w', newline='') as csvfile: 
      csvwriter = csv.writer(csvfile)
      csvwriter.writerow(fields) 
      csvwriter.writerows(rows)
  csvfile.close()
print("[SUCCESS] Computed Class Imbalance successfully")