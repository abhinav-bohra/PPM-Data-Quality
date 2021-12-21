# AUTOGENERATED! DO NOT EDIT! File to edit: 04_prediction_evaluation.ipynb (unless otherwise specified).

__all__ = ['logs', 'ppms', 'isnotebook', 'prepare_for_export', 'get_single_col_dfs']

# Cell
from mppn.imports import *
from mppn.preprocessing import *
from mppn.pipeline import *
from mppn.baselines import *
from mppn.mppn import *

logs=[BPIC_12_Wcomplete, BPIC_12_Wc_const, BPIC_12_Wc_mode_event, BPIC_12_Wc_mode_case]
ppms=[PPM_MPPN]

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
               print_output=print_output,bs=bs,patience=patience,min_delta=min_delta,runs=runs)

    if __name__ == '__main__':
        fire.Fire(command_line)
