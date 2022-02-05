import glob
import os, warnings
import csv, argparse
import pandas as pd
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
for log in logs:
	models = os.listdir(f"{path}/{log}")
	for model in models:
		features = glob.glob(f"{path}/{log}/{model}/features*.csv")
		pivot_file = features[0].split('\\')[1]
		df_pivot = pd.read_csv(f"{path}/{log}/{model}/{pivot_file}")
		for i in range(0,len(features)):
			file = features[i].split('\\')[1]
			df = pd.read_csv(f"{path}/{log}/{model}/{file}")
			if df_pivot.equals(df):
				print("TRUE")
			else:
				print("FALSE: ",file)
		