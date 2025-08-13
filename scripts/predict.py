# scripts/predict.py

#!/usr/bin/env python3
"""
predict.py  -  Iterative forecasting prediction with teacher forcing
Given a features pickle file, predicts the next W frames iteratively and writes a DCD file.
Supports teacher forcing to use ground truth data instead of predictions at each step.

This script loads a trained model from a .pth file and a features_v6.pkl file.
It then iteratively predicts new coordinates by using the model's output as the
input for the next step, with an option to use ground truth data for a specified
probability (teacher forcing).

Usage:
  python predict.py \
    --features features_v6_W15.pkl \
    --model train_v13_model/best.pth \
    --pdb trajectory.pdb \
    --steps 10 \
    --hidden 256 --layers 4 --dropout 0.2 \
    --out predicted_v13.dcd \
    --teacher_forcing 0.5

"""
import os
# Workaround for OpenMP runtime conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import pickle
import numpy as np
import torch
import mdtraj as md
from tqdm import tqdm
from torch_geometric.data import Data
import torch.nn.functional as F
import random

# Import the model from the core directory
from core.model import GraphRNN

# -------------------- Feature Computation --------------------
def compute_features(raw_coords, res_idx, atom_oh, CONTACT_THRESH, device, full_top, atom2res):
    # raw_coords: (W, N, 3), numpy array
    W, N = raw_coords.shape[:2]
    traj = md.Trajectory(raw_coords, topology=full_top)
    rg_seq = md.compute_rg(traj)
    _, phi = md.compute_phi(traj)
    _, psi = md.compute_psi(traj)
    ss_letters = md.compute_dssp(traj, simplified=True)
    cmap = np.zeros((W, N, N), dtype=np.int8)
    for t in range(W):
        d = np.linalg.norm(raw_coords[t][:,None,:] - raw_coords[t][None,:,:], axis=-1)
        cmap[t] = (d < CONTACT_THRESH).astype(np.int8)
    seq = []
    for t in range(W):
        feats = []
        for atom in range(N):
            coord = raw_coords[t, atom]
            r = atom2res[atom]
            phi_t, psi_t = (phi[t, r], psi[t, r]) if (r>=0 and r<phi.shape[1]) else (0.0, 0.0)
            s = ss_letters[t, r] if (r>=0 and r<ss_letters.shape[1]) else 'C'
            ss_oh = [1,0,0] if s=='H' else ([0,1,0] if s=='E' else [0,0,1])
            rg_v = [rg_seq[t]]
            at = atom_oh[atom].cpu().numpy().tolist()
            feats.append(np.concatenate([coord, [phi_t, psi_t], ss_oh, rg_v, at]))
        x = torch.tensor(np.stack(feats), dtype=torch.float32, device=device)
        src, dst = np.where(cmap[t]>0)
        edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)
        data = Data(x=x, edge_index=edge_index)
        data.batch = torch.zeros(x.size(0), dtype=torch.long, device=device)
        seq.append(data)
    return seq

# -------------------- Main --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', required=True)
    parser.add_argument('--model',    required=True)
    parser.add_argument('--pdb',      required=True)
    parser.add_argument('--steps',    type=int, default=1, help='number of windows to predict')
    parser.add_argument('--hidden',   type=int, required=True)
    parser.add_argument('--layers',   type=int, required=True)
    parser.add_argument('--dropout',  type=float, default=0.2)
    parser.add_argument('--out',      default='predicted_v12.dcd')
    parser.add_argument('--compile',  action='store_true')
    parser.add_argument('--edge_thresh', type=float, default=0.8)
    parser.add_argument('--teacher_forcing', type=float, default=0.0, 
                        help='Probability of using ground truth (1.0 = always, 0.0 = never)')
    parser.add_argument('--start_idx', type=int, default=0, 
                        help='Starting window index from the features file')
    parser.add_argument('--stride',       type=int, default=30,
                        help='How many frames to shift between windows (set == window_size for no overlap)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(args.features, 'rb') as f:
        Xc, Xrg, Xcm, Xto, Xss, Yss, Y, res_idx = pickle.load(f)
    W, N = Xc.shape[1], Xc.shape[2]
    
    # Initial window based on start_idx
    start_idx = min(args.start_idx, Xc.shape[0]-1)
    raw_win = Xc[start_idx]

    # subset PDB to backbone atoms only
    sel = [a for idxs in res_idx for a in idxs if a>=0]
    full_traj = md.load(args.pdb)
    sub_traj = full_traj.atom_slice(sel)
    full_top = sub_traj.topology

    atom2res = np.full(N, -1, dtype=int)
    for i, idxs in enumerate(res_idx):
        for a in idxs:
            if a>=0: atom2res[a] = i

    ckpt = torch.load(args.model, map_location=device)
    mean, std = ckpt['mean'].to(device), ckpt['std'].to(device)

    # atom-type one-hot
    atom_type = np.zeros(N, dtype=int)
    for r, idxs in enumerate(res_idx):
        ni, cai, ci, oi = idxs
        if ni>=0:  atom_type[ni]  = 0
        if cai>=0: atom_type[cai] = 1
        if ci>=0:  atom_type[ci]  = 2
        if oi>=0:  atom_type[oi]  = 3
    atom_oh = torch.from_numpy(np.eye(4, dtype=np.float32)[atom_type]).to(device)

    node_dim = 3 + 2 + 3 + 1 + 4
    model = GraphRNN(node_dim, args.hidden, N, 3,
                     n_layers=args.layers, dropout=args.dropout).to(device)
    if args.compile and torch.cuda.is_available():
        model = torch.compile(model)

    sd = ckpt['model']
    clean_sd = {}
    for k,v in sd.items():
        kk = k.replace('_orig_mod.', '')
        if kk.startswith('gcn.'): kk = kk.replace('gcn.', 'gcn_blocks.')
        clean_sd[kk] = v
    model.load_state_dict(clean_sd)
    model.eval()

    preds = []
    max_step = min(args.steps, (Xc.shape[0] - start_idx)) if args.teacher_forcing > 0 else args.steps
    
    for step in tqdm(range(max_step), desc='Predict windows'):
        # Compute features from current window
        seq = compute_features(raw_win, res_idx, atom_oh, args.edge_thresh,
                              device, full_top, atom2res)
        for data in seq:
            data.x[:, :3] = (data.x[:, :3] - mean) / std
        
        # Predict next window
        with torch.no_grad():
            pred_norm, _ = model(seq)
        
        # Denormalize predictions
        pred = (pred_norm.squeeze(0) * std + mean).cpu().numpy()
        preds.append(pred)
        
        # Determine next input window using teacher forcing
        next_idx = start_idx + (step+1)*args.stride
        use_ground_truth = (random.random() < args.teacher_forcing and next_idx < Xc.shape[0])
        
        if use_ground_truth:
            # Use ground truth from dataset
            raw_win = Xc[next_idx]
            print(f"Step {step+1}/{max_step}: Using ground truth (teacher forcing)")
        else:
            # Use model prediction for next step
            raw_win = pred
            print(f"Step {step+1}/{max_step}: Using model prediction")

    all_preds = np.vstack(preds)
    print(all_preds.dtype, all_preds.shape)  
    md.Trajectory(all_preds, topology=full_top).save_dcd(args.out)
    print(f"Saved predicted {all_preds.shape[0]} frames to {args.out}")

if __name__ == '__main__':
    main()