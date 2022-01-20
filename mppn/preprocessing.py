# AUTOGENERATED! DO NOT EDIT! File to edit: 00_preprocessing.ipynb (unless otherwise specified).

__all__ = ['EventLogs', 'import_log', 'drop_long_traces', 'RandomTraceSplitter', 'split_traces', 'PPObj', 'PPProc',
           'Categorify', 'FillStrategy', 'FillMissing', 'Normalize', 'Base_Date_Encodings', 'encode_date',
           'decode_date', 'Datetify', 'MinMax', 'OneHot', 'subsequences_fast', 'PPDset', 'get_dls']

# Cell
import pandas as pd
pd.set_option('display.max_columns', None)
from .imports import *
from imblearn.under_sampling import NearMiss
import logging
ci_flag = False
balancing_technique = None
#Logging
logging.basicConfig(filename="logs/preprocess.log",format='',filemode='w')
logger = logging.getLogger() 
logger.setLevel(logging.DEBUG)
logging.getLogger('numba').setLevel(logging.WARNING)
logger.debug("--Preprocessing Logging--")

# Cell
class EventLogs:
    BPIC_12=Path('./event_logs/BPIC12.csv')
    BPIC_12_A=Path('./event_logs/BPIC12_A.csv')
    BPIC_12_O=Path('./event_logs/BPIC12_O.csv')
    BPIC_12_W=Path('./event_logs/BPIC12_W.csv')
    BPIC_12_Wcomplete=Path('./event_logs/BPIC12_Wc.csv')
    BPIC_13_CP=Path('./event_logs/BPIC13_CP.csv')
    BPIC_15_1=Path('./event_logs/BPIC15_1.csv')
    BPIC_15_2=Path('./event_logs/BPIC15_2.csv')
    BPIC_15_3=Path('./event_logs/BPIC15_3.csv')
    BPIC_15_4=Path('./event_logs/BPIC15_4.csv')
    BPIC_15_5=Path('./event_logs/BPIC15_5.csv') 
    BPIC_17_OFFER=Path('./event_logs/BPIC17_O.csv')
    BPIC_20_RFP=Path('./event_logs/BPIC20_RFP.csv')
    Helpdesk=Path('./event_logs/Helpdesk.csv')
    Mobis=Path('./event_logs/Mobis.csv')
    #New Datasets
    BPIC_12_const=Path('./event_logs/BPIC12_const.csv')
    BPIC_12_mode_event=Path('./event_logs/BPIC12_mode_event.csv')
    BPIC_12_mode_case=Path('./event_logs/BPIC12_mode_case.csv')
    BPIC_12_W_const=Path('./event_logs/BPIC12_W_const.csv')
    BPIC_12_W_mode_event=Path('./event_logs/BPIC12_W_mode_event.csv')
    BPIC_12_W_mode_case=Path('./event_logs/BPIC12_W_mode_case.csv')
    BPIC_12_Wc_const=Path('./event_logs/BPIC12_Wc_const.csv')
    BPIC_12_Wc_mode_event=Path('./event_logs/BPIC12_Wc_mode_event.csv')
    BPIC_12_Wc_mode_case=Path('./event_logs/BPIC12_Wc_mode_case.csv')
    BPIC_13_CP_const=Path('./event_logs/BPIC13_CP_const.csv')
    BPIC_13_CP_mode_event=Path('./event_logs/BPIC13_CP_mode_event.csv')
    BPIC_13_CP_mode_case=Path('./event_logs/BPIC13_CP_mode_case.csv')
    Mobis_const=Path('./event_logs/Mobis_const.csv')
    Mobis_mode_event=Path('./event_logs/Mobis_mode_event.csv')
    Mobis_mode_case=Path('./event_logs/Mobis_mode_case.csv')
    

def import_log(ds): return pd.read_csv(ds,index_col=0)

# Cell
def drop_long_traces(df,max_trace_len=64,event_id='event_id'):
    df=df.drop(np.unique(df[df[event_id]>max_trace_len].index))
    return df

# Cell
def RandomTraceSplitter(split_pct=0.2, seed=None):
    "Create function that splits `items` between train/val with `valid_pct` randomly."
    def _inner(trace_ids):
        o=np.unique(trace_ids)
        np.random.seed(seed)
        rand_idx = np.random.permutation(o)
        cut = int(split_pct * len(o))
        return L(rand_idx[cut:].tolist()),L(rand_idx[:cut].tolist())
    return _inner

# Cell
def split_traces(df,df_name='tmp',test_seed=42,validation_seed=None):
    df=drop_long_traces(df)
    ts=RandomTraceSplitter(seed=test_seed)
    train,test=ts(df.index)
    ts=RandomTraceSplitter(seed=validation_seed,split_pct=0.1)
    train,valid=ts(train)
    return train,valid,test

# Cell
class _TraceIloc:
    "Get/set rows by iloc and cols by name"
    def __init__(self,o): self.o = o
    def __getitem__(self, idxs):
        df = self.o.items
        if isinstance(idxs,tuple):
            rows,cols = idxs
            rows=df.index[rows]
            return self.o.new(df.loc[rows,cols])
        else:
            rows,cols = idxs,slice(None)
            rows=np.unique(df.index)[rows]
            return self.o.new(df.loc[rows])

# Cell
class PPObj(CollBase, GetAttr, FilteredBase):
    "Main Class for Process Prediction"
    _default,with_cont='procs',True
    def __init__(self,df,procs=None,cat_names=None,cont_names=None,date_names=None,y_names=None,splits=None,
                 ycat_names=None,ycont_names=None,inplace=False,do_setup=True):
        if not inplace: df=df.copy()
        if splits is not None: df = df.loc[sum(splits, [])] # Can drop traces
        self.event_ids=df['event_id'].values if hasattr(df,'event_id') else None

        super().__init__(df)

        self.cat_names,self.cont_names,self.date_names=(L(cat_names),L(cont_names),L(date_names))
        self.set_y_names(y_names,ycat_names,ycont_names)

        self.procs = Pipeline(procs)
        self.splits=splits
        if do_setup: self.setup()


    @property
    def y_names(self): return self.ycat_names+self.ycont_names

    def set_y_names(self,y_names,ycat_names=None,ycont_names=None):
        if ycat_names or ycont_names: store_attr('ycat_names,ycont_names')
        else:
            self.ycat_names,self.ycont_names=(L([i for i in L(y_names) if i in self.cat_names]),
                                                L([i for i in L(y_names) if i not in self.cat_names]))
    def setup(self): self.procs.setup(self)
    def subset(self, i): return self.new(self.loc[self.splits[i]]) if self.splits else self
    def __len__(self): return len(np.unique(self.items.index))
    def show(self, max_n=3, **kwargs):
        print('#traces:',len(self),'#events:',len(self.items))
        display_df(self.new(self.all_cols[:max_n]).items)
    def new(self, df):
        return type(self)(df, do_setup=False,
                          **attrdict(self, 'procs','cat_names','cont_names','ycat_names','ycont_names',
                                     'date_names'))
    def process(self): self.procs(self)
    def loc(self): return self.items.loc
    def iloc(self): return _TraceIloc(self)
    def x_names (self): return self.cat_names + self.cont_names
    def all_col_names(self): return ((self.x_names+self.y_names)).unique()
    def transform(self, cols, f, all_col=True):
        if not all_col: cols = [c for c in cols if c in self.items.columns]
        if len(cols) > 0: self[cols] = self[cols].transform(f)
    def new_empty(self): return self.new(pd.DataFrame({}, columns=self.items.columns))
    def subsets(self): return [self.subset(i) for i in range(len(self.splits))] if self.splits else L(self)
properties(PPObj,'loc','iloc','x_names','all_col_names')

def _add_prop(cls, nm):
    @property
    def f(o): return o[list(getattr(o,nm+'_names'))]
    @f.setter
    def fset(o, v): o[getattr(o,nm+'_names')] = v
    setattr(cls, nm+'s', f)
    setattr(cls, nm+'s', fset)

_add_prop(PPObj, 'cat')
_add_prop(PPObj, 'cont')
_add_prop(PPObj, 'ycat')
_add_prop(PPObj, 'ycont')
_add_prop(PPObj, 'y')
_add_prop(PPObj, 'x')
_add_prop(PPObj, 'all_col')

# Cell
class PPProc(InplaceTransform):
    "Base class to write a non-lazy tabular processor for dataframes"
    def setup(self, items=None, train_setup=False): #TODO: properly deal with train_setup
        super().setup(getattr(items,'train',items), train_setup=False)
        #super().setup(items, train_setup=False)

        # Procs are called as soon as data is available
        return self(items.items if isinstance(items,Datasets) else items)

    @property
    def name(self): return f"{super().name} -- {getattr(self,'__stored_args__',{})}"

# Cell
def _apply_cats (voc, add, c):
    if not is_categorical_dtype(c):
        return pd.Categorical(c, categories=voc[c.name][add:]).codes+add
    return c.cat.codes+add #if is_categorical_dtype(c) else c.map(voc[c.name].o2i)

# Cell
class Categorify(PPProc):
    "Transform the categorical variables to something similar to `pd.Categorical`"
    order = 2
    def setups(self, to):
        store_attr(classes={n:CategoryMap(to.items.loc[:,n], add_na=True) for n in to.cat_names}, but='to')
    def encodes(self, to):
        to.transform(to.cat_names, partial(_apply_cats, self.classes, 1))
    def __getitem__(self,k): return self.classes[k]

# Cell
class FillStrategy:
    "Namespace containing the various filling strategies."
    def median  (c,fill): return c.median()
    def constant(c,fill): return fill
    def mode    (c,fill): return c.dropna().value_counts().idxmax()

# Cell
class FillMissing(PPProc):
    order=1
    "Fill the missing values in continuous columns."
    def __init__(self, fill_strategy=FillStrategy.median, add_col=True, fill_vals=None):
        if fill_vals is None: fill_vals = defaultdict(int)
        store_attr()

    def setups(self, dsets):
        missing = pd.isnull(dsets.conts).any()
        store_attr(but='to', na_dict={n:self.fill_strategy(dsets[n], self.fill_vals[n])
                            for n in missing[missing].keys()})
        self.fill_strategy = self.fill_strategy.__name__

    def encodes(self, to):
        missing = pd.isnull(to.conts)
        for n in missing.any()[missing.any()].keys():
            assert n in self.na_dict, f"nan values in `{n}` but not in setup training set"
        for n in self.na_dict.keys():
            to[n].fillna(self.na_dict[n], inplace=True)
            if self.add_col:
                to.loc[:,n+'_na'] = missing[n]
                if n+'_na' not in to.cat_names: to.cat_names.append(n+'_na')

# Cell
class Normalize(PPProc):
    "Normalize with z-score"
    order = 3
    def setups(self, to):
        store_attr(but='to', means=dict(getattr(to, 'train', to).conts.mean()),
                   stds=dict(getattr(to, 'train', to).conts.std(ddof=0)+1e-7))
        return self(to)

    def encodes(self, to): to.conts = (to.conts-self.means) / self.stds
    def decodes(self, to): to.conts = (to.conts*self.stds ) + self.means

# Cell
def _make_date(df, date_field):
    "Make sure `df[date_field]` is of the right date type."
    field_dtype = df[date_field].dtype
    if isinstance(field_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        field_dtype = np.datetime64
    if not np.issubdtype(field_dtype, np.datetime64):
        df[date_field] = pd.to_datetime(df[date_field], infer_datetime_format=True,utc=True)

# Cell
def _secSinceSunNoon(datTimStr):
    dt = pd.to_datetime(datTimStr).dt
    return (dt.dayofweek-1)*24*3600+ dt.hour * 3600 + dt.minute * 60 + dt.second

# Cell
def _secSinceNoon(datTimStr):
    dt = pd.to_datetime(datTimStr).dt
    return dt.hour * 3600 + dt.minute * 60 + dt.second

# Cell
Base_Date_Encodings=['Year', 'Month', 'Day', 'Dayofweek', 'Dayofyear','Elapsed']

# Cell
def encode_date(df, field_name,unit=1e9,date_encodings=Base_Date_Encodings):
    "Helper function that adds columns relevant to a date in the column `field_name` of `df`."
    _make_date(df, field_name)
    field = df[field_name]
    prefix =  re.sub('[Dd]ate$', '', field_name+"_")
    attr = ['Year', 'Month', 'Day', 'Dayofweek', 'Dayofyear', 'Is_month_end', 'Is_month_start',
            'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
    if time: attr = attr + ['Hour', 'Minute', 'Second']
    for n in attr:
        if n in date_encodings: df[prefix + n] = getattr(field.dt, n.lower())
    # Pandas removed `dt.week` in v1.1.10

    if 'secSinceSunNoon' in date_encodings:
        df[prefix+'secSinceSunNoon']=_secSinceSunNoon(field)
    if 'secSinceNoon' in date_encodings:
        df[prefix+'secSinceNoon']=_secSinceNoon(field)
    if 'Week' in date_encodings:
        week = field.dt.isocalendar().week if hasattr(field.dt, 'isocalendar') else field.dt.week
        df.insert(3, prefix+'Week', week)
    mask = ~field.isna()
    elapsed = pd.Series(np.where(mask,field.values.astype(np.int64) // unit,None).astype(float),index=field.index)

    if 'Relative_elapsed' in date_encodings:
        df[prefix+'Relative_elapsed']=elapsed-elapsed.groupby(elapsed.index).transform('min')

    # required to decode!
    if 'Elapsed' in date_encodings: df[prefix+'Elapsed']=elapsed

    df.drop(field_name, axis=1, inplace=True)
    return [],[prefix+i for i in date_encodings]

# Cell
def decode_date(df, field_name,unit=1e9,date_encodings=Base_Date_Encodings):
    df[field_name]=(df[field_name+'_'+'Elapsed'] * unit).astype('datetime64[ns, UTC]')
    for c in date_encodings: del df[field_name+'_'+c]

# Cell
class Datetify(PPProc):
    "Encode dates, "
    order = 0

    def __init__(self, date_encodings=['Relative_elapsed']): self.date_encodings=listify(date_encodings)

    def encodes(self, o):
        for i in o.date_names:
            cat,cont=encode_date(o.items,i,date_encodings=self.date_encodings)
            o.cont_names+=cont
            o.cat_names+=cat
# Todo: Add decoding

# Cell
class MinMax(PPProc):
    order=3

    def setups(self, o):
        store_attr(mins=o.xs.min(),
                   maxs=o.xs.max())

    def encodes(self, o):
        cols=[i+'_minmax' for i in o.x_names]
        o[cols] = o.xs.astype(float)
        o[cols] = ((o.xs-self.mins) /(self.maxs-self.mins))
        o.cont_names=L(cols)
        o.cat_names=L()

# Cell
from sklearn.preprocessing import OneHotEncoder

# Cell
class OneHot(PPProc):
    "Transform the categorical variables to one-hot. Requires Categorify to deal with unseen data."
    order = 3

    def encodes(self, o):
        new_cats=[]
        for c in o.cat_names:
            categories=[range(len(o.procs.categorify[c]))]
            x=o[c].to_numpy()
            ohe = OneHotEncoder(categories=categories)
            enc=ohe.fit_transform(x.reshape(-1, 1)).toarray()
            for i in range(enc.shape[1]):
                new_cat=f'{c}_{i}'
                o.items.loc[:,new_cat]=enc[:,i]
                new_cats.append(new_cat)
        o.cat_names=L(new_cats)

# Cell
def _shift_columns (a,ws=3): return np.dstack(list(reversed([np.roll(a,i) for i in range(0,ws)])))[0]

# Cell
def subsequences_fast(df,event_ids,ws=None,min_ws=64):
    max_trace_len=int(event_ids.max())+1

    if not ws: ws=max_trace_len-1
    elif ws <max_trace_len-1: raise ValueError(f"ws must be greater equal {max_trace_len-1}")
    pad=ws
    ws=max(min_ws,ws)
    trace_start = np.where(event_ids == 0)[0]
    trace_len=np.array([trace_start[i]-trace_start[i-1] for i in range(1,len(trace_start))]+[len(df)-trace_start[-1]])
    tmp=np.stack([_shift_columns(df[i],ws=ws) for i in list(df)])
    idx=[range(trace_start[i],trace_start[i]+trace_len[i]-1) for i in range(len(trace_start))]
    idx=np.array([y for x in idx for y in x])

    res=np.rollaxis(tmp,1)[idx]
    mask=ws-1-event_ids[idx][:,None] > np.arange(res.shape[2])
    res[np.broadcast_to(mask[:,None],res.shape)]=0
    return res,idx+1

# Cell
class PPDset(torch.utils.data.Dataset):
    def __init__(self, inp):
        store_attr('inp')

    def __len__(self): return len(self.inp[0])

    def __getitem__(self, idx):
        xs=tuple([i[idx]for i in self.inp[:-1]])
        ys=tuple([i[idx]for i in self.inp[-1]])
        if len(ys)==1: ys=ys[0]
        return (*xs,ys)

def getStrategy(my_list):
    avg = int(len(my_list)/len(set(my_list)))
    freq = {}
    for items in my_list:
        freq[items] = my_list.count(items)

    ir = round(max(freq.values())/min(freq.values()))
    # logger.debug(f"Imbalance Ratio is {ir}")
    for key in freq:
      if freq[key] > avg:
        freq[key]=avg

    ir = round(max(freq.values())/min(freq.values()))
    # logger.debug(f"Imbalance Ratio is {ir}")
    return freq

def getBalanceData(func,xs,ys):
  x = list(xs)
  for i in range(len(xs)):
    if len(xs[i].size())==2:
      x[i] = torch.unsqueeze(xs[i], dim=1) 
      
  x = torch.cat(x, dim=1)
  x = list(torch.flatten(x, start_dim=1).numpy())
  y = list(ys[0].numpy())

  x_res, y_res = func.fit_resample(x, y)
  indices = torch.tensor(func.sample_indices_)
  
  xs_under = tuple([torch.index_select(xs[i], 0, indices) for i in range(len(xs))])
  ys_under = tuple([torch.index_select(ys[i], 0, indices) for i in range(len(ys))])
  
  r = xs_under[0].size(0)/xs[0].size(0)
  logger.debug(f"{round(1-r,2)*100}% reduction in size by undersampling wrt activity" )
  
  return xs_under, ys_under

def Balance(xs,ys):
  y_labels = list(ys[0].numpy())
  if balancing_technique == "NM":
    logger.debug("---Applying NearMiss---")
    func = NearMiss(n_neighbors=1,sampling_strategy=getStrategy(y_labels))
    return getBalanceData(func,xs,ys)
  elif balancing_technique == "CONN":
    logger.debug("---Applying COndensedNearestNeighbour---")
  elif balancing_technique == "NCR":
    logger.debug("---Applying NeighbourhoodCleaningRule ---")
  else:
    logger.debug("---Invalid Balancing Technique---")
  return xs,ys

# Cell
@delegates(TfmdDL)
def get_dls(ppo:PPObj,windows=subsequences_fast,outcome=False,event_id='event_id',bs=64,**kwargs):
    ds=[]
    for s in ppo.subsets(): #train, dev and test sets
        wds,idx=windows(s.xs,s.event_ids)
        if not outcome: y=s.ys.iloc[idx]
        else: y=s.ys.groupby(s.items.index).transform('last').iloc[idx]
        ycats=tensor(y[s.ycat_names].values).long() #['activity', 'resource'] torch.Size([342, 2])
        yconts=tensor(y[s.ycont_names].values).float() #['timestamp_Relative_elapsed'] torch.Size([342, 1])
        xconts=tensor(wds[:,len(s.cat_names):]).float() # torch.Size([312, 2, 64])
        xcats=tensor(wds[:,:len(s.cat_names)]).long() # torch.Size([312, 1, 64])
        xs=tuple([i.squeeze() for i in [xcats,xconts] if i.shape[1]>0])
        ys=tuple([ycats[:,i] for i in range(ycats.shape[1])])+tuple([yconts[:,i] for i in range(yconts.shape[1])])
        
        # logger.debug("\n---S---")
        # logger.debug(s.iloc[0])
        
        # logger.debug("\n---X---")
        # logger.debug(s.cat_names)
        # logger.debug(xcats.size())
        # logger.debug(s.cont_names)
        # logger.debug(xconts.size())

        # logger.debug("\n---Y---")
        # logger.debug(s.ycat_names)
        # logger.debug(ycats.size())
        # logger.debug(s.ycont_names)
        # logger.debug(yconts.size())
        
        logger.debug("\n--BEFORE--")
        for i in range(len(xs)):
          logger.debug(xs[i].size())
        if ci_flag:
          try:
            # logger.debug("Balancing Dataset...")
            xs,ys = Balance(xs,ys)
          except Exception as E:
            logger.debug(f"\nException Occurred while BALANCING DATASET: {E}\n")
          
        logger.debug("--AFTER--")
        for i in range(len(xs)):
          logger.debug(xs[i].size())
        ds.append(PPDset((*xs,ys)))
        

    return DataLoaders.from_dsets(*ds,bs=bs,**kwargs)
PPObj.get_dls= get_dls