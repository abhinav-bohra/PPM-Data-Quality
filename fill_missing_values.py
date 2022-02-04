import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
warnings.filterwarnings("ignore")

#--------------------------------------------------------------------------------------------
# Helper Function
#--------------------------------------------------------------------------------------------
def groupby_fillna(df, group_col, fill_col):
	  #Group by Activity
	  groups = df.groupby(group_col)
	  #Find most frequent resource in all activity groups
	  mode_by_group = groups[fill_col].transform(lambda x: x.mode()[0] if len(x.mode()) else np.nan)
	  #Fill Nan Resources with the most frequent resource in all activity groups
	  df[fill_col] = df[fill_col].fillna(mode_by_group)
	  return df


#--------------------------------------------------------------------------------------------
# Strategy 1: Replace all NaN values with a constant (0)
#--------------------------------------------------------------------------------------------
def fill_constant(df,root_dir,file,const_val):
	print(f'Filling {file} with constant value {const_val}')
	df['resource'] = df['resource'].fillna(const_val)
	df.to_csv(f'{root_dir}/{file}_const.csv',index=False)


#--------------------------------------------------------------------------------------------
# Strategy 2: Replace all NaN values with event level mode
# Which resource has done that activity the most in the dataset?
#--------------------------------------------------------------------------------------------
def fill_event_mode(df,root_dir,file):
	print(f'Filling {file} with event level mode')
	df = groupby_fillna(df, 'activity', 'resource')
	df.to_csv(f'{root_dir}/{file}_mode_event.csv',index=False)


#--------------------------------------------------------------------------------------------
# Strategy 3: Replace all NaN values with case level mode
# Which resource has done that activity the most in the given case?
#--------------------------------------------------------------------------------------------
def fill_case_mode(df,root_dir,file):
	print(f'Filling {file} with case level mode')
	df_result = pd.DataFrame(columns = df.columns)

	#Group by Activity
	case_groups = df.groupby('trace_id')
	cases = list(case_groups.groups.keys())

	for i in tqdm(range(len(cases))):
	  case_df = case_groups.get_group(cases[i])
	  case_df = groupby_fillna(case_df, 'activity', 'resource')
	  df_result = pd.concat([df_result, case_df])
  
	df_result.to_csv(f'{root_dir}/{file}_mode_case.csv',index=False)


#--------------------------------------------------------------------------------------------
# Main Function
#--------------------------------------------------------------------------------------------

if __name__ == "__main__":
	root_dir = "../event_logs"
	#Datasets with missing values
	datasets = ['BPIC12.csv','BPIC12_W.csv','BPIC12_Wc.csv','BPIC13_CP.csv','Mobis.csv']

	for dataset in datasets:
		try:
			file = dataset.split('.')[0]
			df = pd.read_csv(f'{root_dir}/{dataset}')
			fill_constant(df,root_dir,file,0)
			fill_event_mode(df,root_dir,file)
			fill_case_mode(df,root_dir,file)
		except Exception as E:
			print(dataset)
			print("Exception :", E)
