# AUTOGENERATED! DO NOT EDIT! File to edit: 02_baselines.ipynb (unless otherwise specified).

__all__ = ['maeDurDaysNormalize', 'maeDurDaysMinMax', 'AvgMetric', 'get_metrics', 'multi_loss_sum',
           'Camargo_specialized', 'Camargo_concat', 'Camargo_fullconcat', 'PPM_Camargo_Spezialized',
           'PPM_Camargo_concat', 'PPM_Camargo_fullconcat', 'PPM_RNNwEmbedding', 'Evermann', 'PPM_Evermann', 'Tax_et_al_spezialized',
           'Tax_et_al_shared', 'Tax_et_al_mixed', 'PPM_Tax_Spezialized', 'PPM_Tax_Shared', 'PPM_Tax_Mixed', 'MiDA',
           'PPM_MiDA', 'create_attr_dict', 'attr_list', 'attr_dict']


#------------------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------------------
import warnings, logging, sklearn
warnings.filterwarnings("ignore")
from .imports import *
from .preprocessing import *
from .pipeline import *
from fastai import *
from fastai.text import *
 
#------------------------------------------------------------------------------------------
# Logging
#------------------------------------------------------------------------------------------
logging.basicConfig(filename="baselines.log",format='',filemode='w')
logger = logging.getLogger() 
logger.setLevel(logging.DEBUG)
logging.getLogger('numba').setLevel(logging.WARNING)
logger.debug("--Baselines Logging--")

# Cell
def maeDurDaysNormalize(p,yb,mean=0,std=0,unit=60*60*24):
    """
    Decodes time and converts from seconds to days
    Returns mae
    """
    p=p*std+mean
    yb=yb*std+mean
    return mae(p,yb)/(unit)

# Cell
def maeDurDaysMinMax(p,yb,minn=0,maxx=0,unit=60*60*24):
    """
    Decodes time and converts from seconds to days
    Returns mae
    """

    p=p*(maxx-minn) + minn
    yb=yb*(maxx-minn) + minn
    return mae(p,yb)/(unit)

# Cell
def _accuracy_idx(a,b,i): return accuracy(listify(a)[i],listify(b)[i])

def _precision_idx(a,b,i): 
  pred = listify(a)[i]
  targ = listify(b)[i]
  pred,targ = flatten_check(pred.argmax(dim=-1), targ)
  return sklearn.metrics.precision_score(targ.cpu().detach().numpy(), pred.cpu().detach().numpy(),average='macro')

def _recall_idx(a,b,i): 
  pred = listify(a)[i]
  targ = listify(b)[i]
  pred,targ = flatten_check(pred.argmax(dim=-1), targ)
  return sklearn.metrics.recall_score(targ.cpu().detach().numpy(), pred.cpu().detach().numpy(),average='macro')

def _f1_score_idx(a,b,i): 
  pred = listify(a)[i]
  targ = listify(b)[i]
  pred,targ = flatten_check(pred.argmax(dim=-1), targ)
  return sklearn.metrics.f1_score(targ.cpu().detach().numpy(), pred.cpu().detach().numpy(),average='macro')

# Cell
class AvgMetric(Metric):
    "Average the values of `func` taking into account potential different batch sizes"
    def __init__(self, func):  self.func = func
    def reset(self):           self.total,self.count = 0.,0
    def accumulate(self, learn):
        bs = find_bs(learn.yb)
        self.total += learn.to_detach(self.func(learn.pred, *learn.yb))*bs
        self.count += bs
    @property
    def value(self): return self.total/self.count if self.count != 0 else None
    @property
    def name(self):
        return self.func.__name__ if hasattr(self.func, '__name__') else self.func.func.__name__

# Cell
def get_metrics(o,date_col='timestamp_Relative_elapsed'):

    number_cats=len(o.ycat_names)

    accuracies=[]
    for i in range(number_cats):
        accuracy_func=partial(_accuracy_idx,i=i)
        accuracy_func.__name__= f"acc_{o.ycat_names[i]}"
        avg_accuracy_func=AvgMetric(accuracy_func)
        accuracies.append(avg_accuracy_func)
    
    precisions=[]
    for i in range(number_cats):
        precision_func=partial(_precision_idx,i=i)
        precision_func.__name__= f"pre_{o.ycat_names[i]}"
        avg_precision_func=AvgMetric(precision_func)
        precisions.append(avg_precision_func)

    recalls=[]
    for i in range(number_cats):
        recall_func=partial(_recall_idx,i=i)
        recall_func.__name__= f"rec_{o.ycat_names[i]}"
        avg_recall_func=AvgMetric(recall_func)
        recalls.append(avg_recall_func)

    f1_scores=[]
    for i in range(number_cats):
        f1_score_func=partial(_f1_score_idx,i=i)
        f1_score_func.__name__= f"f1_score_{o.ycat_names[i]}"
        avg_f1_score_func=AvgMetric(f1_score_func)
        f1_scores.append(avg_f1_score_func)
 
    mae_days=None
    if len(o.ycont_names)>0:
        if 'minmax' in o.ycont_names[0]: # Here we expect only one timestamp
            minn,maxx = (o.procs.min_max.mins[date_col],
                         o.procs.min_max.maxs[date_col])
            mae_days=lambda p,y: maeDurDaysMinMax(listify(p)[-1],listify(y)[-1],minn=minn,maxx=maxx)
        else:
            mean,std=(o.procs.normalize.means[date_col],
                      o.procs.normalize.stds[date_col])
            mae_days=lambda p,y: maeDurDaysNormalize(listify(p)[-1],listify(y)[-1],mean=mean,std=std)
        mae_days.__name__='mae_days'
    
    metrics = L(accuracies) + L(precisions) + L(recalls) + L(f1_scores) + mae_days
    return metrics

# Cell
def multi_loss_sum(o,p,y, reduction=None):
    p,y=listify(p),listify(y)
    len_cat,len_cont=len(o.ycat_names),len(o.ycont_names)
    cross_entropies=[F.cross_entropy(p[i],y[i]) for i in range(len_cat)]
    maes=[mae(p[i],y[i]) for i in range(len_cat,len_cat+len_cont)]
    return torch.sum(torch.stack(list(L(cross_entropies)+L(maes))))

# Cell
class Camargo_specialized(torch.nn.Module) :
    def __init__(self, o) :
        super().__init__()
        hidden=25
        vocab_act=len(o.procs.categorify['activity'])
        vocab_res=len(o.procs.categorify['resource'])
        emb_dim_act = int(sqrt(vocab_act))+1
        emb_dim_res = int(sqrt(vocab_res))+1

        self.emb_act = nn.Embedding(vocab_act,emb_dim_act)
        self.emb_res = nn.Embedding(vocab_res,emb_dim_res)

        self.lstm_act = nn.LSTM(emb_dim_act, hidden, batch_first=True, num_layers=2)
        self.lstm_res = nn.LSTM(emb_dim_res, hidden, batch_first=True, num_layers=2)
        self.lstm_tim = nn.LSTM(1, hidden, batch_first=True, num_layers=2)

        self.linear_act = nn.Linear(hidden, vocab_act)
        self.linear_res = nn.Linear(hidden, vocab_res)
        self.linear_tim = nn.Linear(hidden, 1)
    def forward(self, xcat,xcont):
        x_act,x_res,x_tim=xcat[:,0],xcat[:,1],xcont[:,:,None]
        x_act = self.emb_act(x_act)
        x_act,_ = self.lstm_act(x_act)
        x_act = x_act[:,-1]
        x_act = self.linear_act(x_act)
        x_act = F.softmax(x_act,dim=1)

        x_res = self.emb_res(x_res)
        x_res,_ = self.lstm_res(x_res)
        x_res = x_res[:,-1]
        x_res = self.linear_res(x_res)
        x_res = F.softmax(x_res,dim=1)

        x_tim,_ = self.lstm_tim(x_tim)
        x_tim = x_tim[:,-1]
        x_tim = self.linear_tim(x_tim)
        return x_act,x_res,x_tim

# Cell
class Camargo_concat(torch.nn.Module) :
    def __init__(self, o ) :
        super().__init__()
        hidden=25
        vocab_act=len(o.procs.categorify['activity'])
        vocab_res=len(o.procs.categorify['resource'])
        emb_dim_act = int(sqrt(vocab_act))+1
        emb_dim_res = int(sqrt(vocab_res))+1

        self.emb_act = nn.Embedding(vocab_act,emb_dim_act)
        self.emb_res = nn.Embedding(vocab_res,emb_dim_res)

        self.lstm_concat= nn.LSTM(emb_dim_act+emb_dim_res, hidden, batch_first=True, num_layers=1)
        self.lstm_act = nn.LSTM(hidden, hidden, batch_first=True, num_layers=1)
        self.lstm_res = nn.LSTM(hidden, hidden, batch_first=True, num_layers=1)
        self.lstm_tim = nn.LSTM(1, hidden, batch_first=True, num_layers=2)

        self.linear_act = nn.Linear(hidden, vocab_act)
        self.linear_res = nn.Linear(hidden, vocab_res)
        self.linear_tim = nn.Linear(hidden, 1)
    def forward(self, xcat,xcont):
        x_act,x_res,x_tim=xcat[:,0],xcat[:,1],xcont[:,:,None]
        x_act = self.emb_act(x_act)

        x_res = self.emb_res(x_res)
        x_concat=torch.cat((x_act, x_res), 2)
        x_concat,_=self.lstm_concat(x_concat)

        x_act,_ = self.lstm_act(x_concat)
        x_act = x_act[:,-1]
        x_act = self.linear_act(x_act)
        x_act = F.softmax(x_act,dim=1)

        x_res,_ = self.lstm_res(x_concat)
        x_res = x_res[:,-1]
        x_res = self.linear_res(x_res)
        x_res = F.softmax(x_res,dim=1)

        x_tim,_ = self.lstm_tim(x_tim)
        x_tim = x_tim[:,-1]
        x_tim = self.linear_tim(x_tim)
        return x_act,x_res,x_tim

# Cell
class Camargo_fullconcat(torch.nn.Module) :

    def __init__(self, o  ) :
        super().__init__()

        hidden=25
        vocab_act=len(o.procs.categorify['activity'])
        vocab_res=len(o.procs.categorify['resource'])
        emb_dim_act = int(sqrt(vocab_act))+1
        emb_dim_res = int(sqrt(vocab_res))+1

        self.emb_act = nn.Embedding(vocab_act,emb_dim_act)
        self.emb_res = nn.Embedding(vocab_res,emb_dim_res)

        self.lstm_concat= nn.LSTM(emb_dim_act+emb_dim_res+1, hidden, batch_first=True, num_layers=1)
        self.lstm_act = nn.LSTM(hidden, hidden, batch_first=True, num_layers=1)
        self.lstm_res = nn.LSTM(hidden, hidden, batch_first=True, num_layers=1)
        self.lstm_tim = nn.LSTM(hidden, hidden, batch_first=True, num_layers=1)

        self.linear_act = nn.Linear(hidden, vocab_act)
        self.linear_res = nn.Linear(hidden, vocab_res)
        self.linear_tim = nn.Linear(hidden, 1)
    def forward(self, xcat,xcont):
        x_act,x_res,x_tim=xcat[:,0],xcat[:,1],xcont[:,:,None]
        x_act = self.emb_act(x_act)

        x_res = self.emb_res(x_res)
        x_concat=torch.cat((x_act, x_res,x_tim), 2)
        x_concat,_=self.lstm_concat(x_concat)

        x_act,_ = self.lstm_act(x_concat)
        x_act = x_act[:,-1]
        x_act = self.linear_act(x_act)
        x_act = F.softmax(x_act,dim=1)

        x_res,_ = self.lstm_res(x_concat)
        x_res = x_res[:,-1]
        x_res = self.linear_res(x_res)
        x_res = F.softmax(x_res,dim=1)

        x_tim,_ = self.lstm_tim(x_concat)
        x_tim = x_tim[:,-1]
        x_tim = self.linear_tim(x_tim)
        return x_act,x_res,x_tim

# Cell
class PPM_Camargo_Spezialized(PPModel):

    model = Camargo_specialized
    date_names=['timestamp']
    cat_names=['activity','resource']
    y_names=['activity','resource','timestamp_Relative_elapsed']
    cont_names=None
    procs=[Categorify,Datetify,Normalize,FillMissing]

    def setup(self):
        self.o=PPObj(self.log,self.procs,cat_names=self.cat_names,date_names=self.date_names,y_names=self.y_names,
                cont_names=self.cont_names,splits=self.splits)
        
    def next_step_prediction(self): 
        print('Next event prediction training')
        loss=partial(multi_loss_sum,self.o)
        dls=self.o.get_dls(bs=self.bs)
        m=self.model(self.o)
        
        self.res_test,self.res_train,self.res_train_nonOut = self._train_validate(self.o,dls,m,loss=loss,metrics=get_metrics(self.o),
                                                   output_index=[1,2,3,4,5,6,7,8,9])
        
        self.nsp_acc_test,self.nrp_acc_test,self.nsp_pre_test,self.nrp_pre_test,self.nsp_rec_test,self.nrp_rec_test,self.nsp_f1_test,self.nrp_f1_test,self.dtnp_test = self.res_test[0], self.res_test[1], self.res_test[2], self.res_test[3], self.res_test[4], self.res_test[5], self.res_test[6], self.res_test[7], self.res_test[8]
        self.nsp_acc_train,self.nrp_acc_train,self.nsp_pre_train,self.nrp_pre_train,self.nsp_rec_train,self.nrp_rec_train,self.nsp_f1_train,self.nrp_f1_train,self.dtnp_train =self.res_train[0], self.res_train[1], self.res_train[2], self.res_train[3], self.res_train[4], self.res_train[5], self.res_train[6], self.res_train[7], self.res_train[8]
        self.nsp_acc_train_nonOut,self.nrp_acc_train_nonOut,self.nsp_pre_train_nonOut,self.nrp_pre_train_nonOut,self.nsp_rec_train_nonOut,self.nrp_rec_train_nonOut,self.nsp_f1_train_nonOut,self.nrp_f1_train_nonOut,self.dtnp_train_nonOut =self.res_train_nonOut[0], self.res_train_nonOut[1], self.res_train_nonOut[2], self.res_train_nonOut[3], self.res_train_nonOut[4], self.res_train_nonOut[5], self.res_train_nonOut[6], self.res_train_nonOut[7], self.res_train_nonOut[8]

        return [self.nsp_acc_test, self.nsp_pre_test, self.nsp_rec_test, self.nsp_f1_test],\
               [self.nsp_acc_train, self.nsp_pre_train, self.nsp_rec_train, self.nsp_f1_train],\
               [self.nsp_acc_train_nonOut, self.nsp_pre_train_nonOut, self.nsp_rec_train_nonOut, self.nsp_f1_train_nonOut]

    def next_resource_prediction(self):return [self.nrp_acc_test, self.nrp_pre_test, self.nrp_rec_test, self.nrp_f1_test],\
                                              [self.nrp_acc_train, self.nrp_pre_train, self.nrp_rec_train, self.nrp_f1_train],\
                                              [self.nrp_acc_train_nonOut, self.nrp_pre_train_nonOut, self.nrp_rec_train_nonOut, self.nrp_f1_train_nonOut]

    def last_resource_prediction(self): 
        print('Last event prediction training')
        loss=partial(multi_loss_sum,self.o)
        dls=self.o.get_dls(outcome=True,bs=self.bs)
        m=self.model(self.o)
        self.res_test,self.res_train,self.res_train_nonOut = self._train_validate(self.o,dls,m,loss=loss,metrics=get_metrics(self.o),
                                                   output_index=[1,2,3,4,5,6,7,8,9])
        
        self.op_acc_test,self.lrp_acc_test,self.op_pre_test,self.lrp_pre_test,self.op_rec_test,self.lrp_rec_test,self.op_f1_test,self.lrp_f1_test,self.dtlp_test=self.res_test[0], self.res_test[1], self.res_test[2], self.res_test[3], self.res_test[4], self.res_test[5], self.res_test[6], self.res_test[7], self.res_test[8]
        self.op_acc_train,self.lrp_acc_train,self.op_pre_train,self.lrp_pre_train,self.op_rec_train,self.lrp_rec_train,self.op_f1_train,self.lrp_f1_train,self.dtlp_train=self.res_train[0], self.res_train[1], self.res_train[2], self.res_train[3], self.res_train[4], self.res_train[5], self.res_train[6], self.res_train[7], self.res_train[8]
        self.op_acc_train_nonOut,self.lrp_acc_train_nonOut,self.op_pre_train_nonOut,self.lrp_pre_train_nonOut,self.op_rec_train_nonOut,self.lrp_rec_train_nonOut,self.op_f1_train_nonOut,self.lrp_f1_train_nonOut,self.dtlp_train_nonOut=self.res_train_nonOut[0], self.res_train_nonOut[1], self.res_train_nonOut[2], self.res_train_nonOut[3], self.res_train_nonOut[4], self.res_train_nonOut[5], self.res_train_nonOut[6], self.res_train_nonOut[7], self.res_train_nonOut[8]
           
        return [self.lrp_acc_test, self.lrp_pre_test, self.lrp_rec_test, self.lrp_f1_test],\
               [self.lrp_acc_train, self.lrp_pre_train, self.lrp_rec_train, self.lrp_f1_train],\
               [self.lrp_acc_train_nonOut, self.lrp_pre_train_nonOut, self.lrp_rec_train_nonOut, self.lrp_f1_train_nonOut]

    def outcome_prediction(self): return [self.op_acc_test, self.op_pre_test, self.op_rec_test, self.op_f1_test],\
                                         [self.op_acc_train, self.op_pre_train, self.op_rec_train, self.op_f1_train],\
                                         [self.op_acc_train_nonOut, self.op_pre_train_nonOut, self.op_rec_train_nonOut, self.op_f1_train_nonOut]
    def duration_to_next_event_prediction(self): return self.dtnp_test, self.dtnp_train, self.dtnp_train_nonOut
    def duration_to_end_prediction(self): return self.dtlp_test, self.dtlp_train, self.dtlp_train_nonOut
    def activity_suffix_prediction(self): pass
    def resource_suffix_prediction(self): pass

# Cell
class PPM_Camargo_concat(PPM_Camargo_Spezialized):
    model = Camargo_concat

class PPM_Camargo_fullconcat(PPM_Camargo_Spezialized):
    model = Camargo_fullconcat

# Cell
class PPM_RNNwEmbedding(PPModel):
    'Sampe PPM based on RNNwEmbedding'
    model=RNNwEmbedding

    def next_step_prediction(self,outcome=False,col='activity'):
        o=PPObj(self.log,procs=Categorify(),cat_names=col,y_names=col,splits=self.splits)
        dls=o.get_dls(outcome=outcome,bs=self.bs,windows=self.windows)
        m=self.model(o)
        metrics=get_metrics(o)
        output_index = list(range(1, len(metrics)+1))
        return self._train_validate(o,dls,m,metrics=metrics,output_index=output_index)

    def next_resource_prediction(self): return self.next_step_prediction(col='resource')
    def last_resource_prediction(self): return self.next_step_prediction(col='resource',outcome=True)
    def outcome_prediction(self): return self.next_step_prediction(outcome=True)

# Cell
class Evermann(torch.nn.Module) :
    def __init__(self, o) :
        super().__init__()
        vocab_size=len(o.procs.categorify[o.y_names[0]])
        hidden_dim=125
        emb_dim = 5

        self.embeddings = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True, num_layers=2)
        self.linear = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.embeddings(x.squeeze())
        x = self.dropout(x)
        x, _ = self.lstm(x)
        x = self.linear(x[:,-1])
        return F.softmax(x,dim=1)

# Cell
class PPM_Evermann(PPM_RNNwEmbedding):
    model = Evermann

# Cell
class Tax_et_al_spezialized(torch.nn.Module) :
    def __init__(self,o) :
        super().__init__()
        vocab_size=len(o.procs.categorify[o.y_names[0]])
        hidden_dim=125
        self.lstm_act = nn.LSTM(vocab_size, hidden_dim, batch_first=True, num_layers=2)
        self.lstm_tim = nn.LSTM(3, hidden_dim, batch_first=True, num_layers=2)

        self.linear_act = nn.Linear(hidden_dim, vocab_size)
        self.linear_tim = nn.Linear(hidden_dim, 1)

    def forward(self, xcat,xcont):
        x_act = xcat.permute(0,2,1)
        if (len(xcont.squeeze().size())==3):
            x_tim = xcont.squeeze().permute(0,2,1)
        else:
            x_tim = xcont.permute(0,2,1)


        x_act, _ = self.lstm_act(x_act.float())
        x_act=self.linear_act(x_act[:,-1])
        x_act=F.softmax(x_act,dim=1)
        x_tim, _ = self.lstm_tim(x_tim)
        x_tim=self.linear_tim(x_tim[:,-1])
        return x_act,x_tim

# Cell
class Tax_et_al_shared(torch.nn.Module) :
    def __init__(self,o) :
        super().__init__()
        vocab_size=len(o.procs.categorify[o.y_names[0]])
        hidden_dim=125
        self.lstm = nn.LSTM(vocab_size+3, hidden_dim, batch_first=True, num_layers=2)
        self.linear_act = nn.Linear(hidden_dim, vocab_size)
        self.linear_tim = nn.Linear(hidden_dim, 1)


    def forward(self,xcat,xcont):        
        x_act = xcat.permute(0,2,1)
        if (len(xcont.squeeze().size())==3):
            x_tim = xcont.squeeze().permute(0,2,1)
        else:
            x_tim = xcont.permute(0,2,1)

        x_concat=torch.cat((x_act.float(), x_tim), 2)
        x_concat, _ = self.lstm(x_concat)

        x_act=self.linear_act(x_concat[:,-1])
        x_act=F.softmax(x_act,dim=1)

        x_tim=self.linear_tim(x_concat[:,-1])
        return x_act,x_tim

# Cell
class Tax_et_al_mixed(torch.nn.Module) :
    def __init__(self,o,numlayers_shared=3,numlayers_single=3) :
        super().__init__()
        vocab_size=len(o.procs.categorify[o.y_names[0]])
        hidden_dim=125

        self.lstm_act = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, num_layers=numlayers_single)
        self.lstm_tim = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, num_layers=numlayers_single)
        self.lstm = nn.LSTM(vocab_size+3, hidden_dim, batch_first=True, num_layers=numlayers_shared)

        self.linear_act = nn.Linear(hidden_dim, vocab_size)
        self.linear_tim = nn.Linear(hidden_dim, 1)


    def forward(self,xcat,xcont):
        x_act = xcat.permute(0,2,1)
        if (len(xcont.squeeze().size())==3):
            x_tim = xcont.squeeze().permute(0,2,1)
        else:
            x_tim = xcont.permute(0,2,1)

        x_concat=torch.cat((x_act.float(), x_tim), 2)
        x_concat, _ = self.lstm(x_concat)

        x_act, _ = self.lstm_act(x_concat)
        x_act=self.linear_act(x_act[:,-1])
        x_act=F.softmax(x_act,dim=1)

        x_tim, _ = self.lstm_tim(x_concat)
        x_tim=self.linear_tim(x_tim[:,-1])
        return x_act,x_tim

class PPM_Tax_Spezialized(PPModel):

    model = Tax_et_al_spezialized
    date_names=['timestamp']
    cat_names=['activity']
    y_names=['activity','timestamp_Relative_elapsed']
    cont_names=None
    procs=[Categorify,OneHot,Datetify(date_encodings=['secSinceSunNoon','secSinceNoon','Relative_elapsed']),
           Normalize,FillMissing]

    def setup(self):
        self.o=PPObj(self.log,self.procs,cat_names=self.cat_names,date_names=self.date_names,y_names=self.y_names,
                cont_names=self.cont_names,splits=self.splits)

    def next_step_prediction(self): 
        # Next event prediction training
        print('Next event prediction training')
        loss=partial(multi_loss_sum,self.o)
        dls=self.o.get_dls(bs=self.bs)
        m=self.model(self.o)
        self.res_test, self.res_train, self.res_train_nonOut =self._train_validate(self.o,dls,m,loss=loss,metrics=get_metrics(self.o),
                                                   output_index=[1,2,3,4,5])
        
        self.nsp_acc_test, self.nsp_pre_test, self.nsp_rec_test, self.nsp_f1_test, self.dtnp_test = self.res_test[0], self.res_test[1], self.res_test[2], self.res_test[3], self.res_test[4]
        self.nsp_acc_train, self.nsp_pre_train, self.nsp_rec_train, self.nsp_f1_train, self.dtnp_train = self.res_train[0], self.res_train[1], self.res_train[2], self.res_train[3], self.res_train[4]
        self.nsp_acc_train_nonOut, self.nsp_pre_train_nonOut, self.nsp_rec_train_nonOut, self.nsp_f1_train_nonOut, self.dtnp_train_nonOut = self.res_train_nonOut[0], self.res_train_nonOut[1], self.res_train_nonOut[2], self.res_train_nonOut[3], self.res_train_nonOut[4]

        return [ self.nsp_acc_test, self.nsp_pre_test, self.nsp_rec_test, self.nsp_f1_test],\
               [ self.nsp_acc_train, self.nsp_pre_train, self.nsp_rec_train, self.nsp_f1_train],\
               [ self.nsp_acc_train_nonOut, self.nsp_pre_train_nonOut, self.nsp_rec_train_nonOut, self.nsp_f1_train_nonOut]
        
    def next_resource_prediction(self): return [None]*4,[None]*4,[None]*4
    def last_resource_prediction(self): return [None]*4,[None]*4,[None]*4
    def outcome_prediction(self): 
        # Last event prediction training
        print('Last event prediction training')
        loss=partial(multi_loss_sum,self.o)
        dls=self.o.get_dls(outcome=True,bs=self.bs)
        m=self.model(self.o)
        self.res_test, self.res_train, self.res_train_nonOut =self._train_validate(self.o,dls,m,loss=loss,metrics=get_metrics(self.o),
                                                 output_index=[1,2,3,4,5])

        self.op_acc_test, self.op_pre_test, self.op_rec_test, self.op_f1_test, self.dtlp_test = self.res_test[0], self.res_test[1], self.res_test[2], self.res_test[3], self.res_test[4]
        self.op_acc_train, self.op_pre_train, self.op_rec_train, self.op_f1_train, self.dtlp_train = self.res_train[0], self.res_train[1], self.res_train[2], self.res_train[3], self.res_train[4]
        self.op_acc_train_nonOut, self.op_pre_train_nonOut, self.op_rec_train_nonOut, self.op_f1_train_nonOut, self.dtlp_train_nonOut = self.res_train_nonOut[0], self.res_train_nonOut[1], self.res_train_nonOut[2], self.res_train_nonOut[3], self.res_train_nonOut[4]

        return [self.op_acc_test, self.op_pre_test, self.op_rec_test, self.op_f1_test],\
               [self.op_acc_train, self.op_pre_train, self.op_rec_train, self.op_f1_train],\
               [self.op_acc_train_nonOut, self.op_pre_train_nonOut, self.op_rec_train_nonOut, self.op_f1_train_nonOut]

    def duration_to_next_event_prediction(self): return self.dtnp_test,self.dtnp_train,self.dtnp_train_nonOut
    def duration_to_end_prediction(self): return self.dtlp_test,self.dtlp_train,self.dtlp_train_nonOut
    def activity_suffix_prediction(self): pass
    def resource_suffix_prediction(self): pass

# Cell
class PPM_Tax_Shared(PPM_Tax_Spezialized):
    model = Tax_et_al_shared

class PPM_Tax_Mixed(PPM_Tax_Spezialized):
    model = Tax_et_al_mixed

# Cell
class MiDA(Module):
    def __init__(self,o,seq_len=64) :
        super().__init__()
        hidden_dim1=100
        hidden_dim2=100

        out=o.y_names[0]
        emb_szs=[(len(o.procs.categorify[c]),len(o.procs.categorify[c])//2 ) for c in o.cat_names ]
        self.embeds = nn.ModuleList([Embedding(ni, nf) for ni,nf in emb_szs])
        self.n_cont=len(o.cont_names)
        self.n_emb = sum(e.embedding_dim for e in self.embeds)
        self.lstm1=nn.LSTM(self.n_cont+self.n_emb, hidden_dim1, batch_first=True, num_layers=1)

        self.bn_cont = nn.BatchNorm1d(self.n_cont)
        self.lstm2=nn.LSTM(hidden_dim1, hidden_dim2, batch_first=True, num_layers=1)
        #self.bn2=nn.BatchNorm1d(seq_len)
        #self.bn1=nn.BatchNorm1d(seq_len)

        if out in  o.procs.categorify.classes:
            self.lin=nn.Linear(hidden_dim2,len(o.procs.categorify[out]))
            self.is_classifier=True
        else:
            self.lin=nn.Linear(hidden_dim2,1)
            self.is_classifier=False

    def forward(self, x_cat,x_cont):
        if self.n_emb != 0:
            x = [e(x_cat[:,i]) for i,e in enumerate(self.embeds)]
            x = torch.cat(x, 2)
        if self.n_cont != 0:
            if self.n_cont == 1: x_cont=x_cont[:,None]
            if self.bn_cont: x_cont=self.bn_cont(x_cont).transpose(2,1)
            x = torch.cat([x, x_cont], 2) if self.n_emb != 0 else x_cont

        x,_=self.lstm1(x)
        #x= self.bn1(x)
        x,h=self.lstm2(x)
        #x=self.bn2(x[:,-1])
        x=self.lin(x[:,-1])
        if self.is_classifier: x=F.softmax(x,1)
        return x

# Cell

class PPM_MiDA(PPModel):
    model = MiDA

    procs=[Categorify,Normalize,Datetify,FillMissing]

    def _attr_from_dict(self,ds_name):
        if not self.attr_dict: raise AttributeError('attr_dict is required!')

        return (listify(self.attr_dict[self.ds_name]['cat attr']),
                listify(self.attr_dict[self.ds_name]['num attr']),
                listify(self.attr_dict[self.ds_name]['date attr']))

    def setup(self):
        cat_names,cont_names,date_names=self._attr_from_dict(self.ds_name)
        self.o=PPObj(self.log,[Categorify,Normalize,Datetify,FillMissing],
                     cat_names=cat_names,date_names=date_names,cont_names=cont_names,
                     splits=self.splits)

    def next_step_prediction(self,col='activity',outcome=False):
        seq_len=(self.o.items.event_id.max()) # seq_len = max trace len, Todo make it nicer
        self.o.set_y_names(col)
        print(self.o.y_names)
        dls=self.o.get_dls(bs=self.bs,outcome=outcome)
        m=self.model(self.o,seq_len)
        loss=partial(multi_loss_sum,self.o)
        metrics=get_metrics(self.o)
        output_index = list(range(1, len(metrics)+1))
        return self._train_validate(self.o,dls,m,loss=loss,metrics=metrics,output_index=output_index)

    def next_resource_prediction(self):return self.next_step_prediction(outcome=False,col='resource')

    def last_resource_prediction(self): return self.next_step_prediction(outcome=True,col='resource')
    def outcome_prediction(self): return self.next_step_prediction(outcome=True,col='activity')

    def duration_to_next_event_prediction(self):
        return self.next_step_prediction(outcome=False,col='timestamp_Relative_elapsed')

    def duration_to_end_prediction(self):
        return self.next_step_prediction(outcome=True,col='timestamp_Relative_elapsed')

# Cell
def create_attr_dict(attr_list):
    attr_df=pd.DataFrame(attr_list,columns=['name','cat attr','num attr','date attr'])
    attr_df.index=attr_df.name
    attr_df.drop('name',axis=1,inplace=True)
    attr_df.index.name=""
    attr_dict=attr_df.apply(lambda x:(x.str.split(', '))).T.to_dict()
    return attr_dict

# Cell

attr_list=[
    ['BPIC12','activity, resource','AMOUNT_REQ','timestamp'],
    ['BPIC12_A','activity, resource','AMOUNT_REQ','timestamp'],
    ['BPIC12_O','activity, resource','AMOUNT_REQ','timestamp'],
    ['BPIC12_W','activity, resource','AMOUNT_REQ','timestamp'],
    ['BPIC12_Wc','activity, resource','AMOUNT_REQ','timestamp'],    
    ['BPIC15_1','activity, resource',None,'timestamp'],
    ['BPIC15_2','activity, resource',None,'timestamp'],
    ['BPIC15_3','activity, resource',None,'timestamp'],
    ['BPIC15_4','activity, resource',None,'timestamp'],
    ['BPIC15_5','activity, resource',None,'timestamp'],
    ['BPIC15_5','activity, resource',None,'timestamp'],
    ['Mobis','activity, resource, type','cost','timestamp'],
    ['BPIC13_CP','activity, resource, resource country, organization country, organization involved, impact, product, org:role', None,'timestamp'],
    ['Helpdesk','activity, resource',None,'timestamp'],
    ['BPIC17_O','activity, Action, NumberOfTerms, resource','FirstWithdrawalAmount, MonthlyCost, OfferedAmount, CreditScore', 'timestamp'],
    ['BPIC20_RFP','org:role, activity, resource, Project, Task, OrganizationalEntity','RequestedAmount','timestamp'],
    ['BPIC12_const','activity, resource','AMOUNT_REQ','timestamp'],
    ['BPIC12_mode_event','activity, resource','AMOUNT_REQ','timestamp'],
    ['BPIC12_mode_case','activity, resource','AMOUNT_REQ','timestamp'],
    ['BPIC12_W_const','activity, resource','AMOUNT_REQ','timestamp'],
    ['BPIC12_W_mode_event','activity, resource','AMOUNT_REQ','timestamp'],
    ['BPIC12_W_mode_case','activity, resource','AMOUNT_REQ','timestamp'],
    ['BPIC12_Wc_const','activity, resource','AMOUNT_REQ','timestamp'],
    ['BPIC12_Wc_mode_event','activity, resource','AMOUNT_REQ','timestamp'],
    ['BPIC12_Wc_mode_case','activity, resource','AMOUNT_REQ','timestamp'],
    ['BPIC13_CP_const','activity, resource, resource country, organization country, organization involved, impact, product, org:role', None,'timestamp'],
    ['BPIC13_CP_mode_event','activity, resource, resource country, organization country, organization involved, impact, product, org:role', None,'timestamp'],
    ['BPIC13_CP_mode_case','activity, resource, resource country, organization country, organization involved, impact, product, org:role', None,'timestamp'],
    ['Mobis_const','activity, resource, type','cost','timestamp'],
    ['Mobis_mode_event','activity, resource, type','cost','timestamp'],
    ['Mobis_mode_case','activity, resource, type','cost','timestamp']   
]

attr_dict=create_attr_dict(attr_list)
