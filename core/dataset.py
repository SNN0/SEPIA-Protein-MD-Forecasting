
import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')
# --------------------------------------------------------------------------
class ForecastBackboneDataset(Dataset):
    """
    Loads features_v6_backbone.pkl and returns sequence-of-graphs + target coords/ss.
    """
    def __init__(self, pkl_path, edge_thresh=0):
        with open(pkl_path, 'rb') as f:
            Xc, Xrg, Xcm, Xto, Xss, Y_ss, Y, res_idx = pickle.load(f)
        # to tensors
        self.Xc     = torch.tensor(Xc, dtype=torch.float32)
        self.Xrg    = torch.tensor(Xrg, dtype=torch.float32)
        self.Xcm    = torch.tensor(Xcm, dtype=torch.int8)
        self.Xto    = torch.tensor(Xto, dtype=torch.float32)
        self.Xss    = torch.tensor(Xss, dtype=torch.float32)
        self.Y_ss   = torch.tensor(Y_ss, dtype=torch.float32)
        self.Y      = torch.tensor(Y, dtype=torch.float32)
        self.res_idx= res_idx
        self.M, self.W, self.N = self.Xc.shape[:3]
        # normalize coords
        flat = self.Xc.view(-1, 3)
        self.mean = flat.mean(0)
        self.std  = flat.std(0) + 1e-8
        self.edge_thresh = edge_thresh
        # atom-type one-hot (0=N,1=CA,2=C,3=O)
        atom_type = np.zeros(self.N, dtype=int)
        for r, (ni, cai, ci, oi) in enumerate(res_idx):
            if ni>=0: atom_type[ni] = 0
            if cai>=0: atom_type[cai] = 1
            if ci>=0: atom_type[ci] = 2
            if oi>=0: atom_type[oi] = 3
        self.atom_oh = torch.from_numpy(np.eye(4, dtype=np.float32)[atom_type])

    def __len__(self):
        return self.M

    def __getitem__(self, idx):
        # normalized input coords
        coords_in = (self.Xc[idx] - self.mean) / self.std     # (W,N,3)
        coords_t  = (self.Y[idx]  - self.mean) / self.std     # (W,N,3)
        ss_in     = self.Xss[idx]        # (W,N,3)
        ss_t      = self.Y_ss[idx]       # (W,N,3)
        tors_in   = self.Xto[idx]        # (W, n_res-1,2)
        rg_in     = self.Xrg[idx]        # (W,)

        graphs = []
        for t in range(self.W):
            # build node features for frame t
            feats = []
            for atom in range(self.N):
                # torsion: find residue index
                res_i = next((ri for ri, idxs in enumerate(self.res_idx) if atom in idxs), -1)
                if 0 <= res_i < tors_in.shape[1]:
                    phi, psi = tors_in[t, res_i]
                else:
                    phi, psi = 0.0, 0.0
                # features: xyz, torsion, ss, rg, atom-type
                coord = coords_in[t, atom]
                ss_v  = ss_in[t, atom]
                rg_v  = torch.tensor([rg_in[t]], dtype=torch.float32)
                atype = self.atom_oh[atom]
                feats.append(torch.cat([coord, torch.tensor([phi, psi]), ss_v, rg_v, atype]))
            x = torch.stack(feats)
            # edges from contact map
            src, dst = torch.nonzero(self.Xcm[idx, t] > self.edge_thresh, as_tuple=True)
            edge_index = torch.stack([src, dst], dim=0)
            graphs.append(Data(x=x, edge_index=edge_index))

        return graphs, coords_t, ss_t

# collate function
def collate_fn(batch):
    seqs, coords, ss = zip(*batch)
    W = len(seqs[0])
    B = len(seqs)
    # batched graphs per time step
    batched = [Batch.from_data_list([seqs[b][t] for b in range(B)]) for t in range(W)]
    coords = torch.stack(coords)  # (B,W,N,3)
    ss     = torch.stack(ss)      # (B,W,N,3)
    return batched, coords, ss