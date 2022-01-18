# AUTOGENERATED! DO NOT EDIT! File to edit: 04_prediction_evaluation.ipynb (unless otherwise specified).

__all__ = ['logs', 'ppms', 'isnotebook', 'prepare_for_export', 'get_single_col_dfs']

# Cell
from mppn.imports import *
from mppn.preprocessing import *
from mppn.pipeline import *
from mppn.baselines import *
from mppn.mppn import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--exp', default="MV",type=str, help='MV - Missing Values, CI - Class Imbalance')    
parser.add_argument('--save_folder', default="MV",type=str, help='Folder for saving results')    

args = parser.parse_args()
exp = args.exp
save_folder = args.save_folder 
save_dir = f"{save_folder}"
ppms=[PPM_MPPN,PPM_MiDA,PPM_Camargo_concat]

if exp == "MV":
  logs = [EventLogs.BPIC_12_const, EventLogs.BPIC_12_mode_event, EventLogs.BPIC_12_mode_case, \
  EventLogs.BPIC_12_W_const, EventLogs.BPIC_12_W_mode_event, EventLogs.BPIC_12_W_mode_case, \
  EventLogs.BPIC_12_Wc_const, EventLogs.BPIC_12_Wc_mode_event, EventLogs.BPIC_12_Wc_mode_case,\
  EventLogs.BPIC_13_CP_const, EventLogs.BPIC_13_CP_mode_event, EventLogs.BPIC_13_CP_mode_case,\
  EventLogs.Mobis_const, EventLogs.Mobis_mode_event, EventLogs.Mobis_mode_case]
  save_dir = f"01_Missing-Values/{save_folder}"
elif exp == "CI":
  logs=[EventLogs.BPIC_15_5,EventLogs.Helpdesk]
  save_dir = f"02_Class-Imbalance/{save_folder}"
elif exp == "test":
  logs=[EventLogs.Helpdesk]
  save_dir = f"{save_folder}"
  ppms=[PPM_Camargo_concat]
else:
  logs = [EventLogs.BPIC_12, EventLogs.BPIC_12_A, EventLogs.BPIC_12_O, EventLogs.BPIC_12_W, \
  EventLogs.BPIC_12_Wcomplete, EventLogs.BPIC_13_CP, EventLogs.BPIC_15_1, EventLogs.BPIC_15_2, \
  EventLogs.BPIC_15_3, EventLogs.BPIC_15_4, EventLogs.BPIC_15_5, EventLogs. BPIC_17_OFFER, \
  EventLogs.BPIC_20_RFP, EventLogs.Helpdesk, EventLogs.Mobis]


# Cell
import fire

def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False

if not isnotebook():
    from tqdm import tqdm as tqdm_console
    
    def command_line(log_idx=range(len(logs)),ppm_idx=range(len(ppms)),sample=False,store=True, runs=1,
                                   bs=64,print_output=False,patience=3, min_delta=0.005, epoch=20):
        log_sel=L(logs)[log_idx]
        ppm_sel=L(ppms)[ppm_idx]
        
        runner(log_sel,ppm_sel,attr_dict=attr_dict, sample=sample,store=store,epoch=epoch,tqdm=tqdm_console,
               print_output=print_output,bs=bs,patience=patience,min_delta=min_delta,runs=runs,save_dir=save_dir)

    if __name__ == '__main__':
        fire.Fire(command_line)
