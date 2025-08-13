
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------
class WeightedLoss(nn.Module):
    def __init__(self, res_idx, cutoff=0.8, beta=1.0,
                 w_rmsd=1., w_cm=.05, w_rg=.1, w_tor=.1, w_ss=.02, w_wass=.05):
        super().__init__()
        n_idx, ca_idx, c_idx, o_idx = zip(*res_idx)
        self.register_buffer('N_idx',  torch.tensor(n_idx))
        self.register_buffer('CA_idx', torch.tensor(ca_idx))
        self.register_buffer('C_idx',  torch.tensor(c_idx))
        self.cutoff, self.beta = cutoff, beta
        self.set_weights(w_rmsd,w_cm,w_rg,w_tor,w_ss,w_wass)

        # BCE for contacts, CE for SS (coil-favour)
        self.bce = nn.BCEWithLogitsLoss()
        cw = torch.tensor([0.2,0.2,1.0])          # H,E,C
        self.ce  = nn.CrossEntropyLoss(weight=cw)

    # ---------- helper: dynamic weight update ----------
    def set_weights(self, w_rmsd, w_cm, w_rg, w_tor, w_ss, w_wass):
        self.w_rmsd, self.w_cm, self.w_rg = w_rmsd, w_cm, w_rg
        self.w_tor,  self.w_ss, self.w_wass = w_tor,  w_ss,  w_wass

    # ---------- torsion & dihedral utils ----------
    def _torsion(self, c):
        C, N, CA = c[:,:,self.C_idx,:], c[:,:,self.N_idx,:], c[:,:,self.CA_idx,:]
        # φ(i) : C_{i-1}, N_i,   CA_i,  C_i        → length R-1
        phi = self._dihedral(C[:, :, :-1], N[:, :, 1:], CA[:, :, 1:], C[:, :, 1:])
        # ψ(i) : N_i,    CA_i,  C_i,   N_{i+1}     → length R-1
        psi = self._dihedral(N[:, :, :-1], CA[:, :, :-1], C[:, :, :-1], N[:, :, 1:])
        return torch.stack([phi, psi], dim=-1)  # (B,W,R,2)

    @staticmethod
    def _dihedral(p0,p1,p2,p3):
        b0,b1,b2 = p0-p1, p2-p1, p3-p2
        b1 = F.normalize(b1, dim=-1)
        v = b0 - (b0*b1).sum(-1,keepdim=True)*b1
        w = b2 - (b2*b1).sum(-1,keepdim=True)*b1
        x = (v*w).sum(-1)
        y = (torch.cross(b1,v)*w).sum(-1)
        return torch.atan2(y, x)

    # ---------- forward ----------
    def forward(self, pred_c, true_c, ss_logits, ss_true):
        B,W,N,_ = pred_c.shape
        # RMSD
        l_rmsd = F.mse_loss(pred_c, true_c)
        # contact map (scale with 1/β for logit->prob)
        dist = torch.cdist(pred_c.view(B*W,N,3), pred_c.view(B*W,N,3))
        logit = (self.cutoff - dist)/self.beta
        tgt_cm= (torch.cdist(true_c.view(B*W,N,3), true_c.view(B*W,N,3))<self.cutoff).float()
        l_cm  = self.bce(logit.view(B*W,-1), tgt_cm.view(B*W,-1))
        # Rg
        cm_p = pred_c.mean(2); cm_t = true_c.mean(2)
        rg_p = torch.sqrt(((pred_c-cm_p.unsqueeze(2)).pow(2).sum(-1).sum(-1)/N)+1e-8)
        rg_t = torch.sqrt(((true_c-cm_t.unsqueeze(2)).pow(2).sum(-1).sum(-1)/N)+1e-8)
        l_rg = F.mse_loss(rg_p, rg_t)
        # torsion
        tor_p, tor_t = self._torsion(pred_c), self._torsion(true_c)
        l_tor = (F.mse_loss(torch.sin(tor_p), torch.sin(tor_t)) +
                 F.mse_loss(torch.cos(tor_p), torch.cos(tor_t)))
        # SS
        labels = ss_true.argmax(-1).view(-1)
        l_ss   = self.ce(ss_logits.view(B*W*N,-1), labels)
        # Wasserstein (sorted CA-CA distances)
        pd = dist.view(B*W,-1); td = tgt_cm.view(B*W,-1)
        l_wass = torch.mean(torch.abs(torch.sort(pd,dim=1)[0] -
                                      torch.sort(td,dim=1)[0]))
        # total
        total = (self.w_rmsd*l_rmsd + self.w_cm*l_cm + self.w_rg*l_rg +
                 self.w_tor*l_tor   + self.w_ss*l_ss + self.w_wass*l_wass)
        comps = torch.tensor([l_rmsd,l_cm,l_rg,l_tor,l_ss,l_wass], device=total.device)
        return total, comps