import time, torch
from common import *
threads(2)
ds = make_ds(F_EXP)
for width in ([2,1,1],[2,5,1]):
    m = KAN(width=width, grid=3, k=3, seed=0, auto_save=False)
    for G in [3,100,1000]:
        if G!=3: m = m.refine(G)
        t=time.time(); r = m.fit(ds, opt='LBFGS', steps=10, update_grid=False)
        print(width, G, 'sec/step', round((time.time()-t)/10,3), 'test', r['test_loss'][-1], 'params', kan_params(m), flush=True)
mm = make_mlp([2,50,50,1],0); t=time.time(); h=train_mlp(mm,ds,steps=10); print('mlp', mlp_params(mm), (time.time()-t)/10)
