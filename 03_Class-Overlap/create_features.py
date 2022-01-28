import torch
import pandas as pd
from mppn.imports import *

obj = pd.read_pickle(r'test/models/run0/Helpdesk/Camargo_concat/dls.pickle')
train = obj[0].dataset
dev = obj[1].dataset
test = obj[2].dataset
ds = train + dev + test

features = list()
for row in ds:
  x_cats = tuple([t for t in row[0]])
  x_conts = tuple([row[1]])
  new_row = (torch.cat(x_cats + x_conts)).cpu().detach().numpy()
  features.append(new_row)

cols = [f"act_{i}" for i in range(0,64)] + [f"res_{i}" for i in range(0,64)] + [f"time_{i}" for i in range(0,64)]
df = pd.DataFrame(features, columns = cols)
case_len =[int(torch.count_nonzero(row[0][0])) for row in ds]
df.insert(0, "case_len", case_len, True)
df.to_csv('class_overlap.csv', index=False)