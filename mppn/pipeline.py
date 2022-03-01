# AUTOGENERATED! DO NOT EDIT! File to edit: 01_pipeline.ipynb (unless otherwise specified).

__all__ = ['RNNwEmbedding', 'HideOutput', 'training_loop', 'train_validate', 'PPModel', 'get_ds_name',
           'Performance_Statistic', 'runner']

#------------------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------------------
from .imports import *
from .preprocessing import *
import torch, pickle, logging
import pandas as pd

#------------------------------------------------------------------------------------------
# Logging
#------------------------------------------------------------------------------------------
logging.basicConfig(filename="pipeline.log",format='',filemode='w')
logger = logging.getLogger() 
logger.setLevel(logging.DEBUG)
logging.getLogger('numba').setLevel(logging.WARNING)
logger.debug("--Pipeline Logging--")

#------------------------------------------------------------------------------------------
# UDFs
#------------------------------------------------------------------------------------------
def getCaselen(X):
    #X[-1] will be duration feature
    if X[-1][0].shape[0] == 64:
        case_len =[int(torch.count_nonzero(row)) for row in X[-1]]
    else:
        case_len =[int(torch.count_nonzero(row[-1])) for row in X[-1]]
    return case_len


#--------------------------------------------
#Function to save predictions for given task
#--------------------------------------------
def save_preds(preds_,model,store_path,task_name):
    preds = list(preds_)
    preds[0] = getCaselen(preds_[0])
    if "Camargo" in model:
        preds1 = (preds[0],preds[1][0],preds[2][0])
        preds2 = (preds[0],preds[1][1],preds[2][1])
        preds3 = (preds[0],preds[1][2],preds[2][2])
        if "next" in task_name:
            with open(f'{store_path}/preds-next_step_prediction.pickle', 'wb') as f1:
                pickle.dump(preds1, f1)
            with open(f'{store_path}/preds-next_resource_prediction.pickle', 'wb') as f2:
                pickle.dump(preds2, f2)
            with open(f'{store_path}/preds-duration_to_next_event_prediction.pickle', 'wb') as f3:
                pickle.dump(preds3, f3)
        else:
            with open(f'{store_path}/preds-outcome_prediction.pickle', 'wb') as f1:
                pickle.dump(preds1, f1)
            with open(f'{store_path}/preds-last_resource_prediction.pickle', 'wb') as f2:
                pickle.dump(preds2, f2)
            with open(f'{store_path}/preds-duration_to_end_prediction.pickle', 'wb') as f3:
                pickle.dump(preds3, f3)
    elif "Tax" in model:
        preds1 = (preds[0],preds[1][0],preds[2][0])
        preds2 = (preds[0],preds[1][1],preds[2][1])
        if "next" in task_name:
            with open(f'{store_path}/preds-next_step_prediction.pickle', 'wb') as f1:
                pickle.dump(preds1, f1)
            with open(f'{store_path}/preds-duration_to_next_event_prediction.pickle', 'wb') as f2:
                pickle.dump(preds2, f2)
        else:
            with open(f'{store_path}/preds-outcome_prediction.pickle', 'wb') as f1:
                pickle.dump(preds1, f1)
            with open(f'{store_path}/preds-duration_to_end_prediction.pickle', 'wb') as f2:
                pickle.dump(preds2, f2)
    else:
        with open(f'{store_path}/preds-{task_name}.pickle', 'wb') as f:
            pickle.dump(preds, f)


#--------------------------------------------
#Function to save features & targets
#--------------------------------------------
def save_features_targets(obj, store_path, o, task_name):
    train = obj[0].dataset
    dev = obj[1].dataset
    test = obj[2].dataset
    ds = train + dev + test
    store_path = str(store_path)
    model_name = store_path.split('/')[-1]

    features = list()
    targets = list()
    for row in ds:
        #Features
        x = (list(row))[:-1]
        try:
          if(len(x.size())==0):#scalar
            x = x.unsqueeze(0)
        except:
          pass
        x_ = [torch.flatten(t) for t in x]
        ftr = torch.hstack(x_)
        ftr = ftr.detach().cpu().numpy()
        features.append(ftr)
        #Targets
        y = (list(row))[-1]
        try:
          if(len(y.size())==0):#scalar
            y = y.unsqueeze(0)
        except:
          pass
        y_ = [torch.flatten(t) for t in y]
        tar = torch.hstack(y_)
        tar = tar.detach().cpu().numpy()
        targets.append(tar)

    # Saving Features
    num_features = len(features[0])
    feat_size = num_features//len(o.x_names)
    ft_cols = []
    for x_name in o.x_names:
        if x_name in o.cat_names:
            varType = "CAT"
        elif x_name in o.cont_names:
            varType = "CONT"
        else:
            varType = "OTHER"
        ft_cols = ft_cols + [f"{varType}_{x_name}_{i}" for i in range(0,feat_size)]

    df = pd.DataFrame(features, columns = ft_cols)

    if ds[0][-2].shape[0] == 64:
      logger.debug(f"{store_path} has only 1 continous col")
      case_len =[int(torch.count_nonzero(row[-2])) for row in ds]
    else:
      logger.debug(f"{store_path} has more than 1 continous col. Size:{ds[0][-2].shape}")
      case_len =[int(torch.count_nonzero(row[-2][0])) for row in ds]
    df.insert(0, "case_len", case_len, True)
    df.to_csv(f'{store_path}/features.csv', index=False)
    # df.to_csv(f'{store_path}/features-{task_name}.csv', index=False)
    logger.debug(f"Features saved at - {store_path}/features-{task_name}.csv")

    # Saving Targets
    num_targets = len(targets[0])
    targ_size = num_targets//len(o.y_names)
    tg_cols = []
    for y_name in o.y_names:
        if y_name in o.ycat_names:
            varType = "CAT"
        elif y_name in o.ycont_names:
            varType = "CONT"
        else:
            varType = "OTHER"
        tg_cols = tg_cols + [f"{varType}_{y_name}_{i}" for i in range(0,targ_size)]
    df = pd.DataFrame(targets, columns = tg_cols)

    # Handling Camargo and Tax targets
    if "Camargo" in model_name:
        df1 = pd.DataFrame(df.iloc[:,0])
        df2 = pd.DataFrame(df.iloc[:,1])
        df3 = pd.DataFrame(df.iloc[:,2])
        if "next" in task_name:
            df1.to_csv(f'{store_path}/targets-next_step_prediction.csv', index=False)
            df2.to_csv(f'{store_path}/targets-next_resource_prediction.csv', index=False)
            df3.to_csv(f'{store_path}/targets-duration_to_next_event_prediction.csv', index=False)
            logger.debug(f"Targets saved at - {store_path}/targets-next_step_prediction.csv")
            logger.debug(f"Targets saved at - {store_path}/targets-next_resource_prediction.csv")
            logger.debug(f"Targets saved at - {store_path}/targets-duration_to_next_event_prediction.csv")
        else:
            df1.to_csv(f'{store_path}/targets-outcome_prediction.csv', index=False)
            df2.to_csv(f'{store_path}/targets-last_resource_prediction.csv', index=False)
            df3.to_csv(f'{store_path}/targets-duration_to_end_prediction.csv', index=False)
            logger.debug(f"Targets saved at - {store_path}/targets-outcome_prediction.csv")
            logger.debug(f"Targets saved at - {store_path}/targets-last_resource_prediction.csv")
            logger.debug(f"Targets saved at - {store_path}/targets-duration_to_end_prediction.csv")
    elif "Tax" in model_name:
        df1 = pd.DataFrame(df.iloc[:,0])
        df2 = pd.DataFrame(df.iloc[:,1])
        if "next" in task_name:
            df1.to_csv(f'{store_path}/targets-next_step_prediction.csv', index=False)
            df2.to_csv(f'{store_path}/targets-duration_to_next_event_prediction.csv', index=False)
            logger.debug(f"Targets saved at - {store_path}/targets-next_step_prediction.csv")
            logger.debug(f"Targets saved at - {store_path}/targets-duration_to_next_event_prediction.csv")
        else:
            df1.to_csv(f'{store_path}/targets-outcome_prediction.csv', index=False)
            df2.to_csv(f'{store_path}/targets-duration_to_end_prediction.csv', index=False)
            logger.debug(f"Targets saved at - {store_path}/targets-outcome_prediction.csv")
            logger.debug(f"Targets saved at - {store_path}/targets-duration_to_end_prediction.csv")
    else:
        df.to_csv(f'{store_path}/targets-{task_name}.csv', index=False)
        logger.debug(f"Targets saved at - {store_path}/targets-{task_name}.csv")


# Cell
class RNNwEmbedding(torch.nn.Module) :
    def __init__(self,o) :
        super().__init__()
        vocab_size=len(o.procs.categorify[o.y_names[0]])
        emb_size = int(sqrt(vocab_size))+1
        hidden_six = 20
        self.emb = nn.Embedding(vocab_size,emb_size)
        self.rnn = nn.RNN(emb_size, hidden_six, batch_first=True, num_layers=2)
        self.linear = nn.Linear(hidden_six, vocab_size)

    def forward(self, x):
        x = self.emb(x)
        x,_ = self.rnn(x)
        x = x[:,-1]
        x = self.linear(x)
        x = F.softmax(x,dim=1)
        return x

# Cell
class HideOutput:
    'A utility function that hides all outputs in a context'
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

#----------------------------------------------------------------------------------------------------
# Save Features & Targets
#----------------------------------------------------------------------------------------------------

# Cell
def training_loop(learn,epoch,print_output,lr_find):
    '''
    Basic training loop that uses learning rate finder and one cycle training.
    See fastai docs for more information
    '''
    if lr_find:
        lr=np.median([learn.lr_find(show_plot=print_output)[0] for i in range(5)])
        learn.fit_one_cycle(epoch,float(lr))
    else: learn.fit(epoch,0.01)

# Cell
def train_validate(o,dls,m,metrics=accuracy,loss=F.cross_entropy,epoch=20,print_output=True,model_dir=".",lr_find=True,
                   output_index=1,patience=3,min_delta=0.005,show_plot=True,store_path='tmp',model_name='.model'):
    '''
    Trains a model on the training set with early stopping based on the validation loss.
    Afterwards, applies it to the test set.
    '''
    cbs = [CudaCallback,EarlyStoppingCallback(monitor='valid_loss',min_delta=min_delta, patience=patience)]
    learn=Learner(dls, m, path=store_path, model_dir=model_dir, loss_func=loss ,metrics=metrics,cbs=cbs)

    task_name=model_name
    model = str(store_path).split('/')[-1]

    logger.debug("-----"*10)
    logger.debug(f"TASK NAME: {task_name}")
    logger.debug("-----"*10)

    if print_output:
        training_loop(learn,epoch,show_plot,lr_find=lr_find)
        preds=tuple(learn.get_preds(dl=dls[2], with_input=True))
        save_features_targets(dls, store_path, o, task_name)
        save_preds(preds,model,store_path,task_name)
        return learn.validate(dl=dls[2])[output_index]

    else:
        with HideOutput(),learn.no_bar():
            training_loop(learn,epoch,show_plot,lr_find=lr_find)
            preds=tuple(learn.get_preds(dl=dls[2], with_input=True))
            save_features_targets(dls, store_path, o, task_name)
            save_preds(preds,model,store_path,task_name)          
            return learn.validate(dl=dls[2])[output_index]


# Cell
# Todo: Add Logging
class PPModel():

    def __init__(self,log,ds_name,splits,store=None,bs=64,print_output=False,patience=3,min_delta=0.005,
                 attr_dict=None,windows=partial(subsequences_fast,min_ws=0),epoch=20,sample=False,
                 train_validate=train_validate):
        store_attr('log,ds_name,splits,attr_dict,windows,epoch,bs,print_output,min_delta,patience,store')
        self.lr_find=True
        if sample:
            self.lr_find=False
            traces=self.splits[0]
            self.splits=traces[:60],traces[60:80],traces[80:100]
            self.bs=64
            self.epoch=1

    def evaluate(self):
        if not self.print_output:
            with HideOutput(): return self.__evaluate()
        else: return self.__evaluate()
    def __evaluate(self):
        print(self.ds_name,self.get_name())
        self.setup()

        print('next_step_prediction')
        nsp_acc,nsp_pre,nsp_rec,nsp_f1=self.next_step_prediction()

        print('next_resource_prediction')
        nrp_acc,nrp_pre,nrp_rec,nrp_f1=self.next_resource_prediction()

        print('last_resource_prediction')
        lrp_acc,lrp_pre,lrp_rec,lrp_f1=self.last_resource_prediction()

        print('outcome_prediction')
        op_acc,op_pre,op_rec,op_f1=self.outcome_prediction()

        print('duration_to_next_event_prediction')
        dtnep=self.duration_to_next_event_prediction()

        print('duration_to_end_prediction')
        dtep=self.duration_to_end_prediction()

        print('activity_suffix_prediction')
        asp=self.activity_suffix_prediction()

        print('resource_suffix_prediction')
        rsp=self.resource_suffix_prediction()
        
        return nsp_acc,nsp_pre,nsp_rec,nsp_f1,nrp_acc,nrp_pre,nrp_rec,nrp_f1,lrp_acc,lrp_pre,lrp_rec,lrp_f1,op_acc,op_pre,op_rec,op_f1,dtnep,dtep,asp,rsp

    def _train_validate(self,o,dls,m,metrics=accuracy,loss=F.cross_entropy,output_index=1):
        store,model_name='tmp','.model'
        if self.store:
            ins_stack=inspect.stack()
            model_name=str(ins_stack[2][3]) if str(ins_stack[2][3])!='__evaluate' else str(ins_stack[1][3])
            store=self.store/self.ds_name/self.get_name()
        return train_validate(o,dls,m,metrics=metrics,loss=loss,output_index=output_index, #Only change these
                              epoch=self.epoch,print_output=self.print_output,patience=self.patience,
                              min_delta=self.min_delta,show_plot=False,store_path=store,model_name=model_name,
                              lr_find=self.lr_find)
    def setup(self): pass
    def get_name(self): return self.__class__.__name__.replace('PPM_',"")
    def next_step_prediction(self): pass
    def next_resource_prediction(self): pass
    def last_resource_prediction(self): pass
    def outcome_prediction(self): pass
    def duration_to_next_event_prediction(self): pass
    def duration_to_end_prediction(self): pass
    def activity_suffix_prediction(self): pass
    def resource_suffix_prediction(self): pass

# Cell
def get_ds_name(url): return(url.stem) # Utility function, that gets the name of a dataset

# Cell
from datetime import datetime
import inspect
from tqdm.notebook import tqdm

# Cell
class Performance_Statistic():
    'Creates a results dataframe, that shows the performance of all models on all datasets on all tasks.'
    def __init__(self):
        self.df = pd.DataFrame(
        columns=['Dataset', 'Model', 'Balancing Technique', 'Next Step Acc','Next Step Pre','Next Step Rec','Next Step F1',\
         'Next Resource Acc','Next Resource Pre','Next Resource Rec','Next Resource F1', \
         'Last Resource Acc','Last Resource Pre','Last Resource Rec','Last Resource F1', \
         'Outcome Acc','Outcome Pre','Outcome Rec','Outcome F1', \
         'Next relative Timestamp', 'Duration to Outcome', 'Activity Suffix', 'Resource Suffix'])
    def update(self,model_performance): self.df.loc[len(self.df)] = model_performance
    def to_df(self):
        return self.df

# Cell
def _store_path(save_dir,results_dir=Path('./')):
    'Creates a new folder to store results'
    #now = datetime.now()
    #current_time = now.strftime("%Y-%m-%d_%H:%M:%S")
    #results_dir=results_dir/current_time
    results_dir = results_dir/save_dir
    results_dir.mkdir(exist_ok=True)
    return results_dir

# Cell
@delegates(PPModel)
def runner(dataset_urls,ppm_classes,save_dir,balancing_technique,store=True,runs=1,sample=False,validation_seed=None,test_seed=42,tqdm=tqdm,
           **kwargs):
    store_path= _store_path(save_dir) if store else None
    '''
    Runs a number of process prediction models PPModel on a number of datasets for multiple runs.
    Stores results in ./tmp folder.
    '''
    i=0
    results=[]
    #Loop over num_runs
    for r in tqdm(range(runs),desc='Runs'):
        performance_statistic = Performance_Statistic()
        db=tqdm(range(len(dataset_urls)),leave=False)
        #Loop over datasets
        for i in db:
            db.set_description(get_ds_name(dataset_urls[i]))
            ds= dataset_urls[i]
            log=import_log(ds)
            # log=log[:350]
            ds_name=get_ds_name(ds)
            splits=split_traces(log,ds_name,validation_seed=validation_seed,test_seed=test_seed)
            # if store:
            #     with open(store_path/f'run{r}_{ds_name}_splits.pickle', "wb") as output_file:
            #         pickle.dump(splits, output_file)
            mb=tqdm(range(len(ppm_classes)),leave=False)
            #Loop over models
            for j in mb:
                mb.set_description(ppm_classes[j].__name__.replace('PPM_',""))
                ppm_class=ppm_classes[j]
                model_path=store_path/'models'/f"run{r}" if store else None
                model=ppm_class(log,ds_name,splits,store=model_path,sample=sample,**kwargs)
                logger.debug("*"*50)
                logger.debug(ds_name)
                logger.debug(model.get_name())
                logger.debug("*"*50)
                model_performance = model.evaluate()
                logger.debug(f"model_performance: {model_performance}\n\n")
                model_performance = [ds_name, model.get_name(),balancing_technique,*model_performance]
                performance_statistic.update(model_performance)
                [ds_name, model.get_name(),*model_performance]

        df = performance_statistic.to_df()
        results.append(df)
        if store: df.to_csv(store_path/f"run_{r}_results.csv", index=False)
    return results if len(results)>1 else results[0]
