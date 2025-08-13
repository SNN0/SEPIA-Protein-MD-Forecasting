
import os, math, argparse, pickle, yaml, numpy as np, torch
import torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import modular components
from core.dataset import ForecastBackboneDataset, collate_fn
from core.model import GraphRNN, GCNBlock
from core.losses import WeightedLoss
# --------------------------------------------------------------------------
# train loop
def train_loop(args):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds  = ForecastBackboneDataset(args.features)
    loader = DataLoader(ds,batch_size=args.batch,shuffle=True,
                        collate_fn=collate_fn,num_workers=args.workers,pin_memory=True)
    # model
    node_dim = ds[0][0][0].x.size(1)
    model = GraphRNN(node_dim,args.hidden,ds.N,ds.Y_ss.size(-1),
                     n_layers=args.layers,dropout=args.dropout,
                     grad_ckpt=args.grad_ckpt).to(dev)
    if args.compile and torch.cuda.is_available():
        model = torch.compile(model)

    crit = WeightedLoss(ds.res_idx, w_rmsd=args.w_rmsd, w_cm=args.w_cm,
                        w_rg=args.w_rg, w_tor=args.w_tor,
                        w_ss=args.w_ss, w_wass=args.w_wass).to(dev)
    opt  = optim.Adam(model.parameters(), lr=args.lr)
    scaler, best = GradScaler(), math.inf

    for epoch in range(1, args.epochs+1):
        # ---- curriculum & ramp weights ----------
        if epoch <= 20:
            ramp = 0.0
        elif epoch <= 40:
            ramp = (epoch - 20) / 20.0
        else:
            ramp = 1.0

        w_cm   = args.w_cm   * ramp
        w_wass = args.w_wass * ramp

        # RG ramp: 30–40 epoch arası lineer, sonrasında 1.0
        ramp_rg = 0.0 if epoch <= 30 else min((epoch - 30) / 10.0, 1.0)
        w_rg    = args.w_rg * ramp_rg
        crit.set_weights(args.w_rmsd, w_cm, w_rg,
                        args.w_tor, args.w_ss, w_wass)

        model.train(); tot=0; comp_sum=torch.zeros(6,device=dev)
        for seq, tgt_c, tgt_ss in tqdm(loader, desc=f"Ep{epoch:3d}"):
            tgt_c, tgt_ss = tgt_c.to(dev), tgt_ss.to(dev)
            seq = [b.to(dev) for b in seq]
            opt.zero_grad(set_to_none=True)
            with autocast():
                pred_c, ss_log = model(seq)
                loss, comps = crit(pred_c, tgt_c, ss_log, tgt_ss)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tot += loss.item(); comp_sum += comps
        avg, comp_avg = tot/len(loader), comp_sum/len(loader)
        print(f"E{epoch:3d} tot={avg:.4f}"
              f" RMSD={comp_avg[0]:.3f} CM={comp_avg[1]:.3f}"
              f" RG={comp_avg[2]:.3f} Tor={comp_avg[3]:.3f}"
              f" SS={comp_avg[4]:.3f} WASS={comp_avg[5]:.3f}")

        if avg < best:
            best = avg
            torch.save({'model':model.state_dict(),'mean':ds.mean,'std':ds.std},
                       os.path.join(args.output,'best.pth'))

# --------------------------------------------------------------------------
# argparser & main
if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--features', required=True)
    p.add_argument('--output',   default='train_v13_model')
    p.add_argument('--epochs',   type=int, default=120)
    p.add_argument('--batch',    type=int, default=4)
    p.add_argument('--hidden',   type=int, default=512)
    p.add_argument('--layers',   type=int, default=6)
    p.add_argument('--lr',       type=float, default=1e-3)
    p.add_argument('--dropout',  type=float, default=0.2)
    p.add_argument('--workers',  type=int, default=4)
    p.add_argument('--compile',  action='store_true')
    p.add_argument('--grad_ckpt',action='store_true')
    # ---- hedef ağırlık değerleri ----
    p.add_argument('--w_rmsd', type=float, default=1.0)
    p.add_argument('--w_cm',   type=float, default=0.05)
    p.add_argument('--w_rg',   type=float, default=0.10)
    p.add_argument('--w_tor',  type=float, default=0.10)
    p.add_argument('--w_ss',   type=float, default=0.02)
    p.add_argument('--w_wass', type=float, default=0.05)
    args=p.parse_args(); os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output,'config.yaml'),'w') as f:
        yaml.dump(vars(args),f)
    train_loop(args)