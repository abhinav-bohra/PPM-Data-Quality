# AUTOGENERATED! DO NOT EDIT! File to edit: 01_pipeline.ipynb (unless otherwise specified).

__all__ = ['RNNwEmbedding', 'HideOutput', 'training_loop', 'train_validate', 'PPModel', 'get_ds_name',
           'PPM_RNNwEmbedding', 'Performance_Statistic', 'runner']

# Cell
from .imports import *
from .preprocessing import *
import pickle
import logging
 
# Create and configure logger
logging.basicConfig(filename="pipeline.log",format='%(asctime)s %(message)s',filemode='w')
# Creating an object
logger = logging.getLogger() 
# Setting the threshold of logger to DEBUG
logger.setLevel(logging.DEBUG)
# Test messages
logger.debug("--Logging--")

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
def train_validate(dls,m,metrics=[accuracy,F1Score],loss=F.cross_entropy,epoch=20,print_output=True,model_dir=".",lr_find=True,
                   output_index=1,patience=3,min_delta=0.005,show_plot=True,store_path='tmp',model_name='.model'):
    '''
    Trains a model on the training set with early stopping based on the validation loss.
    Afterwards, applies it to the test set.
    '''
    logger.debug("--Train Validate--")
    logger.debug(f" DLS is {dls}")
    logger.debug(f" Model is {m}")
    logger.debug(f" Metrics is {metrics}")

    logger.debug("-----------------")
    cbs = [CudaCallback,
      EarlyStoppingCallback(monitor='valid_loss',min_delta=min_delta, patience=patience),
      SaveModelCallback(fname=model_name)
      ]
    learn=Learner(dls, m, path=store_path, model_dir=model_dir, loss_func=loss ,metrics=metrics,cbs=cbs)

    if print_output:
        training_loop(learn,epoch,show_plot,lr_find=lr_find)
        return learn.validate(dl=dls[2])[output_index]
    else:
        with HideOutput(),learn.no_bar(),learn.no_logging():
            training_loop(learn,epoch,show_plot,lr_find=lr_find)
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
        nsp_acc,nsp_pre=self.next_step_prediction()

        print('next_resource_prediction')
        nrp_acc,nrp_pre=self.next_resource_prediction()

        print('last_resource_prediction')
        lrp_acc,lrp_pre=self.last_resource_prediction()

        print('outcome_prediction')
        op_acc,op_pre=self.outcome_prediction()

        print('duration_to_next_event_prediction')
        dtnep=self.duration_to_next_event_prediction()

        print('duration_to_end_prediction')
        dtep=self.duration_to_end_prediction()

        print('activity_suffix_prediction')
        asp=self.activity_suffix_prediction()

        print('resource_suffix_prediction')
        rsp=self.resource_suffix_prediction()
        
        return nsp_acc, nsp_pre, nrp_acc, nrp_pre, lrp_acc, lrp_pre, op_acc, op_pre, dtnep, dtep, asp, rsp

    def _train_validate(self,dls,m,metrics=[accuracy,F1Score],loss=F.cross_entropy,output_index=1):
        store,model_name='tmp','.model'
        if self.store:
            ins_stack=inspect.stack()
            model_name=str(ins_stack[2][3]) if str(ins_stack[2][3])!='__evaluate' else str(ins_stack[1][3])
            store=self.store/self.ds_name/self.get_name()
        return train_validate(dls,m,metrics=metrics,loss=loss,output_index=output_index, #Only change these
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
class PPM_RNNwEmbedding(PPModel):
    'Sampe PPM based on RNNwEmbedding'
    model=RNNwEmbedding

    def next_step_prediction(self,outcome=False,col='activity'):
        o=PPObj(self.log,procs=Categorify(),cat_names=col,y_names=col,splits=self.splits)
        dls=o.get_dls(outcome=outcome,bs=self.bs,windows=self.windows)
        m=self.model(o)
        return self._train_validate(dls,m)

    def next_resource_prediction(self): return self.next_step_prediction(col='resource')
    def last_resource_prediction(self): return self.next_step_prediction(col='resource',outcome=True)
    def outcome_prediction(self): return self.next_step_prediction(outcome=True)

# Cell
from datetime import datetime
import inspect
from tqdm.notebook import tqdm

# Cell
class Performance_Statistic():
    'Creates a results dataframe, that shows the performance of all models on all datasets on all tasks.'
    def __init__(self):
        self.df = pd.DataFrame(
        columns=['Dataset', 'Model', 'Next Step Acc','Next Step Pre', 'Next Resource Acc','Next Resource Pre', \
         'Last Resource Acc','Last Resource Pre', 'Outcome Acc','Outcome Pre', \
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
    results_dir.mkdir()
    return results_dir

# Cell
@delegates(PPModel)
def runner(dataset_urls,ppm_classes,save_dir,store=True,runs=1,sample=False,validation_seed=None,test_seed=42,tqdm=tqdm,
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
            ds_name=get_ds_name(ds)
            splits=split_traces(log,ds_name,validation_seed=validation_seed,test_seed=test_seed)
            if store:
                with open(store_path/f'run{r}_{ds_name}_splits.pickle', "wb") as output_file:
                    pickle.dump(splits, output_file)
            mb=tqdm(range(len(ppm_classes)),leave=False)
            #Loop over models
            for j in mb:
                mb.set_description(ppm_classes[j].__name__.replace('PPM_',""))
                ppm_class=ppm_classes[j]
                model_path=store_path/'models'/f"run{r}" if store else None
                model=ppm_class(log,ds_name,splits,store=model_path,sample=sample,**kwargs)
                model_performance = model.evaluate()
                logger.debug(model_performance)
                model_performance = [ds_name, model.get_name(),*model_performance]
                performance_statistic.update(model_performance)
                [ds_name, model.get_name(),*model_performance]

        df = performance_statistic.to_df()
        results.append(df)
        if store: df.to_csv(store_path/f"run_{r}_results.csv")
    return results if len(results)>1 else results[0]
