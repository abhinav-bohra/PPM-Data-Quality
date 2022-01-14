import os,csv
import pandas as pd
import numpy as np
from tqdm import tqdm
from imbalance_degree import imbalance_degree

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
  metrics = [size,num_classes,frequency_dist,empirical_dist,ir, multi_group, minority_percent]
  metrics = metrics + id_scores
  return metrics

#--------------------------------------------------------------------------------------------
# Main Function
#--------------------------------------------------------------------------------------------

if __name__ == "__main__":
	datasets = list(os.listdir('../event_logs/'))
	cat_cols = ['activity','resource']

	fields = ["Dataset","Column","Size of Dataset","Num Classes","Frequency Distribution","Empirical Distribution","Imbalance Ratio",\
	 "Multi-Minority/Majority","Minority Class", "Euclidean distance",	"Chebyshev distance",\
	 "Kullback Leibler divergence","Hellinger distance","Total variation distance","Chi-square divergence"]
												
	rows = list()
	for dataset in datasets:
	  df = pd.read_csv(f"event_logs/{dataset}")
	  df =  df.dropna() 
	  for col in cat_cols:
	    row = [dataset,col] + (class_imbalance_analysis(df,col))
	    rows.append(row)

	with open("class_imbalance_results.csv", 'w') as csvfile: 
	    csvwriter = csv.writer(csvfile)
	    csvwriter.writerow(fields) 
	    csvwriter.writerows(rows)
	csvfile.close()