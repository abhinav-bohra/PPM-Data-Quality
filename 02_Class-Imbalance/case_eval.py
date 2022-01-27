import pandas as pd

obj = pd.read_pickle(r'data1.pickle')
x_input = obj[0]
seq_len =[int(torch.count_nonzero(ele[0])) for ele in x_input[0]]
num_cats = x_input[0].size(1)
num_conts = x_input[1].size(1)

preds =[torch.argmax(obj[1][i],dim=1) for i in range(0,num_cats)]
preds.append(torch.squeeze(obj[1][num_cats]))
preds = tuple(preds)
targs = obj[2]
preds = tuple([pred.tolist() for pred in preds])
targs = tuple([targ.tolist() for targ in targs])

data = {
    'seq_len': seq_len,
    'pred_act': preds[0],
    'pred_res': preds[1],
    'pred_time': preds[2],
    'targ_act': targs[0],
    'targ_res': targs[1],
    'targ_time': targs[2],
}
df = pd.DataFrame(data)


#Group by seq_len and eval