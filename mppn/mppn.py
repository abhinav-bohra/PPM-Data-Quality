# AUTOGENERATED! DO NOT EDIT! File to edit: 03_mppn.ipynb (unless otherwise specified).

__all__ = ['BaseMPPN', 'MPPNClassifier', 'MPPNMultitask', 'load_checkpoint', 'mppn_pretraining_model', 'model_urls',
           'mppn_representation_learning_model', 'mppn_fine_tuning_model', 'gaf_transform',
           'mppn_get_output_attributes', 'PPM_MPPN']

#------------------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------------------
import logging
from .imports import *
from .preprocessing import *
from .pipeline import *
from .baselines import *

#------------------------------------------------------------------------------------------
# Logging
#------------------------------------------------------------------------------------------ 
logging.basicConfig(filename="mppn.log",format='',filemode='w')
logger = logging.getLogger() 
logger.setLevel(logging.DEBUG)
logging.getLogger('numba').setLevel(logging.WARNING)
logger.debug("--MPPN Logging--")


class BaseMPPN(nn.Module):

    def __init__(self, num_perspectives, feature_size=64, output_dim=128):
        super(BaseMPPN, self).__init__()
        self.feature_size = feature_size
        self.output_dim = output_dim
        self.mode = 99
        self.stop_training = False

        if self.mode == 1:
            self.num_perspectives = 1
        else:
            self.num_perspectives = num_perspectives

        self.CNN = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=4, padding=2),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(32, 64, kernel_size=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=2),
            nn.Conv2d(64, 64, kernel_size=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=64, out_channels=self.feature_size, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, stride=2),
            # nn.Conv2d(128, self.feature_size, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.MLP = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(self.num_perspectives * self.feature_size, self.num_perspectives * self.feature_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(self.num_perspectives * self.feature_size, self.num_perspectives * int(self.feature_size)),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.1),
            # nn.Linear(self.num_perspectives*self.feature_size, self.num_perspectives*int(self.feature_size)),
            # nn.ReLU(inplace=True),
            nn.Linear(self.num_perspectives * int(self.feature_size), self.output_dim),
        )

    def forward(self, x):
        x = x.transpose(0, 1)

        view_pool = []

        for v in x:
            """size of v: [batch_size, 3, img_height, img_width]"""
            #print(v.size())
            """Get features from GAFs using CNN"""
            v = self.CNN(v)
            #print(v.size())
            """size of v: [batch_size, feature_size, ?, ?], last two should be 1, 1"""
            """Reduce dimensions from 4 to 2 (first is batchsize)"""
            v = v.view(v.size(0), self.feature_size)
            view_pool.append(v)

        pooled_view = view_pool[0]
        """Max-pooling of all views"""
        if self.mode == 1:
            for i in range(1, len(view_pool)):
                pooled_view = torch.max(pooled_view, view_pool[i])
        else:
            """Concatenate features from all perspectives"""
            pooled_view = torch.cat(view_pool, dim=1)
            #print(pooled_view.size())

        """Get representation from MLP"""
        representation = self.MLP(pooled_view)

        return representation

    def count_parameters(self):

        param_CNN = sum(p.numel() for p in self.MLP.parameters() if p.requires_grad)
        param_MLP = sum(p.numel() for p in self.MLP.parameters() if p.requires_grad)
        params = param_CNN + param_MLP
        return params

# Cell

class MPPNClassifier(BaseMPPN):
    """
    Extends Base MPPN with one classification layer.
    """

    def __init__(self, num_perspectives, num_classes, feature_size=64, output_dim=128,with_softmax=True):
        super().__init__(num_perspectives, feature_size=feature_size, output_dim=output_dim)
        self.with_softmax=num_classes!=1
        self.classification = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(output_dim, num_classes)
        )

    def forward(self, x):
        representation = BaseMPPN.forward(self, x)
        classifier_output = self.classification(representation)
        if self.with_softmax:
            classifier_output=F.log_softmax(classifier_output,dim=1)
        return classifier_output

# Cell

class MPPNMultitask(BaseMPPN):
    """
    Extends the MPPNClassifier with multiple heads to predict multiple outputs at once.
    """
    def __init__(self, num_perspectives, output_attr, feature_size=64, representation_dim=128):
        super().__init__(num_perspectives, feature_size=feature_size, output_dim=representation_dim)
        del self.MLP

        self.output_attr = output_attr
        self.representation_dim = representation_dim

        """Get the output dimension of the CNN"""
        self.CNN_out_dim = self.CNN[-3].out_channels

        self.MLP = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(self.num_perspectives * self.feature_size, self.num_perspectives * int(self.feature_size/2)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(self.num_perspectives * int(self.feature_size/2), self.representation_dim),
        )

        """Dynamically create heads for each attribute to predict"""
        self.output_attr = self.output_attr
        self.heads = nn.ModuleList()
        for attr_name, output_dim in self.output_attr.items():
            #print("Head", attr_name)
            self.heads.append(self.create_head(output_dim))


    def forward(self, x):

        x = x.transpose(0, 1)
        view_pool = []

        for v in x:
            """size of v: [batch_size, 3, img_height, img_width]"""
            # print(v.size())
            """Get features from GAFs using CNN"""
            v = self.CNN(v)
            # print(v.size())
            """size of v: [batch_size, feature_size, ?, ?], last two should be 1, 1"""
            """Reduce dimensions from 4 to 2 (first is batchsize)"""
            v = v.view(v.size(0), self.feature_size)

            view_pool.append(v)

        pooled_view = view_pool[0]
        """Max-pooling of all views"""
        if self.mode == 1:
            for i in range(1, len(view_pool)):
                pooled_view = torch.max(pooled_view, view_pool[i])

        else:
            """Concatenate features from all perspectives"""
            pooled_view = torch.cat(view_pool, dim=1)
            # print(pooled_view.size())

        shared = self.MLP(pooled_view)
        outputs = []

        """Predict each attribute"""
        for head in self.heads:
            if head[-1].out_features > 1:
                outputs.append(F.log_softmax(head(shared), dim=1))
            else:
                outputs.append(head(shared))

        return outputs

    def create_head(self, num_classes):
        """Create a head, i.e. a subnetwork to predict a certain attribute in multi-task fashion"""
        head = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(self.representation_dim, self.representation_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.representation_dim, num_classes)
        )

        return head

    def count_parameters(self):

        param_CNN = sum(p.numel() for p in self.MLP.parameters() if p.requires_grad)
        param_MLP = sum(p.numel() for p in self.MLP.parameters() if p.requires_grad)
        param_heads = sum(p.numel() for head in self.heads for p in head.parameters() if p.requires_grad)
        params = param_CNN + param_MLP + param_heads

        return params

# Cell
def load_checkpoint(path, filename):
    loadpath = os.path.join(path, filename + '_checkpoint.pth.tar')
    assert os.path.isfile(loadpath), 'Error: no checkpoint file found!'
    checkpoint = torch.load(loadpath)
    return checkpoint

# Cell
import torch.utils.model_zoo as model_zoo
model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

def mppn_pretraining_model(pretrained=False, **kwargs):
    """Returns a model either pretrained as alexnet or on GAF images."""
    model = MPPNClassifier(**kwargs)

    pretrained_model = "alexnet"

    if pretrained:
        if pretrained_model == "alexnet":
            print("Loading Alexnet to train MPPNs CNN from scratch")
            pretrained_dict = model_zoo.load_url(model_urls['alexnet'])
            model_dict = model.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                               k in model_dict and v.shape == model_dict[k].shape}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            model.load_state_dict(model_dict)

        elif pretrained_model == "MPPN_GAF":
            print("Load pretrained MPPN trained with GAFs on variant classification")
            checkpoint = load_checkpoint(os.path.join(root_dir(), "data", "ML", "checkpoint"),
                                         filename="MPPN_gaf_pretrained")
            best_model = checkpoint["model"]
            best_model.load_state_dict(checkpoint['state_dict'])

            model.CNN = best_model.CNN

    return model

# Cell
def mppn_representation_learning_model(pretrained, num_perspectives, output_attr, feature_size=64, representation_dim=128):
    """Returns a model for representation learning (multitask). CNN is pretrained on GAF images"""
    model = MPPNMultitask(num_perspectives, output_attr, feature_size, representation_dim)

    if pretrained:
        print("Load pretrained MPPN trained with GAFs on variant classification")
        #checkpoint = load previously trained model on GAF images
        best_model = checkpoint["model"]
        best_model.load_state_dict(checkpoint['state_dict'])

        model.CNN = best_model.CNN
    else:
        alexnet_mppn = mppn_pretraining_model(pretrained=True, num_perspectives=num_perspectives, num_classes=1)
        model.CNN = alexnet_mppn.CNN

    return model

# Cell
def mppn_fine_tuning_model(representation_model, num_perspectives, num_classes):
    """
    Fine-tune a MPPN model that has been trained as representation model on a specific task

    Parameters
    ----------
    dataset: Specifies the dataset, for which the model was trained before.

    Returns
    -------
    """
    model = MPPNClassifier(num_perspectives, num_classes=num_classes)
    model.CNN = representation_model.CNN
    model.MLP = representation_model.MLP
    return model

# Cell

from pyts.image import GramianAngularField
from PIL import Image

def _gaf_loop(e,transformer):
        inp,y=e
        inp[inp>1]=1
        inp[inp<0]=0
        inp=inp*2-1
        x=torch.stack(
            tuple(_gaf_attr(inp[:,i],transformer) for i in range(inp.shape[1]))
        ).transpose(0,1)
        x=x[:,:,None].expand(-1,-1,3,-1,-1)
        return x,y

def _gaf_attr(x,transformer):
    try:

        x = transformer.transform(x)
    except ValueError as e:
        print(x)
        raise e
    x=tensor(x).cuda()
    x = x * 255
    #x=to_rgb(x)
    return x


class gaf_transform(ItemTransform):
    def __init__(self,gs=64):
        self.transformer=GramianAngularField(image_size=gs,sample_range=None, method="s", overlapping=True)

    def encodes(self,e): return _gaf_loop(e,self.transformer)


# Cell
def mppn_get_output_attributes(o):
    output_attributes = {i:len(o.procs.categorify[i]) for i in o.ycat_names }
    for i in o.ycont_names: output_attributes[i]=1
    return output_attributes


# Cell
import copy
import sklearn
import numpy as np

# Cell
def precision(a,b): 
  pred = (a.cpu().detach())
  targ = (b.cpu().detach())
  pred,targ = flatten_check(pred.argmax(axis=-1), targ)
  return sklearn.metrics.precision_score(targ.numpy(), pred.numpy(),average='macro')

def recall(a,b): 
  pred = (a.cpu().detach())
  targ = (b.cpu().detach())
  pred,targ = flatten_check(pred.argmax(axis=-1), targ)
  return sklearn.metrics.recall_score(targ.numpy(), pred.numpy(),average='macro')

def f1(a,b): 
  pred = (a.cpu().detach())
  targ = (b.cpu().detach())
  pred,targ = flatten_check(pred.argmax(axis=-1), targ)
  return sklearn.metrics.f1_score(targ.numpy(), pred.numpy(),average='macro')

class PPM_MPPN(PPModel):


    def _attr_from_dict(self,ds_name):
        if not self.attr_dict: raise AttributeError('attr_dict is required!')

        return (listify(self.attr_dict[self.ds_name]['cat attr']),
                listify(self.attr_dict[self.ds_name]['num attr']),
                listify(self.attr_dict[self.ds_name]['date attr']))

    def setup(self):
        def act_acc(p,y): return accuracy(p[0],y[0])
        def act_pre(p,y): return precision(p[0],y[0])
        def act_rec(p,y): return recall(p[0],y[0])
        def act_f1(p,y): return f1(p[0],y[0])
        def res_acc(p,y): return accuracy(p[1],y[1])
        def res_pre(p,y): return precision(p[1],y[1])
        def res_rec(p,y): return recall(p[1],y[1])
        def res_f1(p,y): return f1(p[1],y[1])
        cat_names,cont_names,date_names=self._attr_from_dict(self.ds_name)
        self.o=PPObj(self.log,[Categorify,Datetify,FillMissing,MinMax],
                     cat_names=cat_names,date_names=date_names,cont_names=cont_names,
                     y_names=['activity','resource','timestamp_Relative_elapsed'],
                     splits=self.splits)
        cont_names_list = self.o.cont_names        
        self.o.cont_names=['timestamp_Relative_elapsed']
        norm=Normalize()
        self.o.procs.add(norm,self.o)
        self.mean=norm.means['timestamp_Relative_elapsed']
        self.std=norm.stds['timestamp_Relative_elapsed']
        # self.o.cont_names=L(['activity_minmax','resource_minmax','timestamp_Relative_elapsed_minmax'])
        self.o.cont_names=L(cont_names_list)
        logger.debug(f"MPPN CONT NAMES: {self.o.cont_names}")
        logger.debug(f"MPPN CAT NAMES: {self.o.cat_names}")          
        self.output_attributes=mppn_get_output_attributes(self.o)
        self.pretrain = mppn_representation_learning_model(False, len(self.o.cont_names), self.output_attributes)
        dls=self.o.get_dls(after_batch=gaf_transform,bs=self.bs)
        loss=partial(multi_loss_sum,self.o)
        time_metric=lambda p,y: maeDurDaysNormalize(listify(p)[-1],listify(y)[-1],mean=self.mean,std=self.std)
        self._train_validate(self.o,dls,self.pretrain,loss=loss,metrics=[act_acc,act_pre, act_rec, act_f1, res_acc, res_pre, res_rec, res_f1,time_metric])

    def next_step_prediction(self,col='activity',outcome=False):
        pretrain=copy.deepcopy(self.pretrain)
        m = mppn_fine_tuning_model(pretrain, len(self.output_attributes), self.output_attributes[col])
        self.o.ycat_names,self.o.ycont_names=L(col),L()
        dls=self.o.get_dls(after_batch=gaf_transform,bs=self.bs,outcome=outcome)
        loss=partial(multi_loss_sum,self.o)
        metrics=get_metrics(self.o)
        return self._train_validate(self.o,dls,m,loss=loss,metrics=metrics,output_index=[1,2,3,4])

    def next_resource_prediction(self):return self.next_step_prediction(outcome=False,col='resource')
    def last_resource_prediction(self): return self.next_step_prediction(outcome=True,col='resource')
    def outcome_prediction(self): return self.next_step_prediction(outcome=True)
    def duration_to_next_event_prediction(self,outcome=False,col='timestamp_Relative_elapsed'):
        pretrain=copy.deepcopy(self.pretrain)
        time=partial(maeDurDaysNormalize,mean=self.mean,std=self.std)
        m = mppn_fine_tuning_model(pretrain, len(self.output_attributes), self.output_attributes[col])
        self.o.ycat_names,self.o.ycont_names=L(),L(col)
        dls=self.o.get_dls(after_batch=gaf_transform,bs=self.bs,outcome=outcome)
        xb,yb=dls.one_batch()
        return self._train_validate(self.o,dls,m,loss=mae,metrics=time)
    def duration_to_end_prediction(self):return self.duration_to_next_event_prediction(outcome=True)
    def activity_suffix_prediction(self): pass
    def resource_suffix_prediction(self): pass
