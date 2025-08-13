
import argparse
import pickle
import numpy as np
import mdtraj as md
from tqdm import tqdm
import multiprocessing as mp

# Distance threshold for contact map in nanometers (0.8 nm ≈ 8 Å)
CONTACT_THRESH = 0.55  # nm

# Simplified secondary structure map
SS3 = dict(H=[1, 0, 0], E=[0, 1, 0], C=[0, 0, 1])

def residue_backbone_idx(sel_idxs, topology):
    sel_set = set(sel_idxs)
    res_idx = []
    for res in topology.residues:
        atoms = {a.name: a.index for a in res.atoms}
        idxs = [atoms.get(name, None) for name in ('N','CA','C','O')]
        pos = [sel_idxs.index(ix) if ix in sel_set else -1 for ix in idxs]
        res_idx.append(tuple(pos))
    return res_idx


def calc_contact_map_batch(coords_chunk, thresh):
    n_frames, n_atoms = coords_chunk.shape[0], coords_chunk.shape[1]
    cmaps = np.zeros((n_frames, n_atoms, n_atoms), dtype=np.int8)
    for i in range(n_frames):
        dmat = np.linalg.norm(coords_chunk[i][:,None,:] - coords_chunk[i][None,:,:], axis=-1)
        cmaps[i] = (dmat < thresh).astype(np.int8)
    return cmaps


def process_windows(coords, rg, cmap, tors, ss_onehot, window_size):
    # number of sliding windows for forecasting
    n_frames = len(coords) - 2*window_size + 1

    # allocate input windows
    Xc      = np.zeros((n_frames, window_size) + coords.shape[1:], dtype=np.float32)
    Xrg     = np.zeros((n_frames, window_size), dtype=np.float32)
    Xcm     = np.zeros((n_frames, window_size) + cmap.shape[1:], dtype=np.int8)
    Xto     = np.zeros((n_frames, window_size) + tors.shape[1:], dtype=np.float32)
    Xss     = np.zeros((n_frames, window_size) + ss_onehot.shape[1:], dtype=np.float32)

    # allocate forecasting targets for next window_size frames
    Y       = np.zeros((n_frames, window_size) + coords.shape[1:], dtype=np.float32)
    Y_ss    = np.zeros((n_frames, window_size) + ss_onehot.shape[1:], dtype=np.float32)

    for i in range(n_frames):
        sl_in  = slice(i,               i + window_size)
        sl_out = slice(i + window_size, i + 2*window_size)

        # inputs
        Xc[i]   = coords[sl_in]
        Xrg[i]  = rg[sl_in]
        Xcm[i]  = cmap[sl_in]
        Xto[i]  = tors[sl_in]
        Xss[i]  = ss_onehot[sl_in]

        # targets: subsequent window
        Y[i]    = coords[sl_out]
        Y_ss[i] = ss_onehot[sl_out]

    return Xc, Xrg, Xcm, Xto, Xss, Y_ss, Y


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pdb',     required=True)
    p.add_argument('--dcd',     required=True)
    p.add_argument('--window',  type=int, default=30)
    p.add_argument('--step',    type=int, default=1)
    p.add_argument('--start',   type=int, default=0)
    p.add_argument('--stop',    type=int)
    p.add_argument('--atoms',   choices=['backbone','all'], default='backbone')
    p.add_argument('--out',     default='features_v6_backbone.pkl')
    p.add_argument('--batch_size',           type=int, default=50)
    p.add_argument('--max_medoid_frames',    type=int, default=100)
    p.add_argument('--n_jobs',    type=int, default=mp.cpu_count())
    args = p.parse_args()

    # load topology and trajectory
    top = md.load_topology(args.pdb)
    sel_str = 'name N or name CA or name C or name O' if args.atoms=='backbone' else 'protein'
    sel_idxs = top.select(sel_str)
    traj = md.load(args.dcd, top=args.pdb, atom_indices=sel_idxs)

    # frame indices
    all_idx = list(range(args.start, args.stop or traj.n_frames, args.step))
    # medoid selection for alignment
    def find_medoid(traj, frames, max_f):
        if len(frames) > max_f:
            stride = len(frames) // max_f
            samp = frames[::stride][:max_f]
        else:
            samp = frames
        sub = traj.slice(samp)
        rmsd_mat = np.zeros((len(samp), len(samp)))
        for i in range(len(samp)):
            rmsd_mat[i] = md.rmsd(sub, sub, i)
        return samp[int(np.argmin(rmsd_mat.sum(1)))]

    med = find_medoid(traj, all_idx, args.max_medoid_frames)
    ref = traj.slice([med]); ref.save_pdb('medoid_ref.pdb')
    traj = traj.superpose(ref)

    # raw features
    coords = traj.xyz[all_idx].astype(np.float32)
    rg     = md.compute_rg(traj)[all_idx].astype(np.float32)

    # contact maps
    cmap = np.zeros((len(all_idx), traj.n_atoms, traj.n_atoms), dtype=np.int8)
    for i in tqdm(range(0, len(all_idx), args.batch_size), 'CM batches'):
        batch = coords[i:i+args.batch_size]
        cmap[i:i+batch.shape[0]] = calc_contact_map_batch(batch, CONTACT_THRESH)

    # torsions
    _, phi = md.compute_phi(traj); _, psi = md.compute_psi(traj)
    tors = np.stack([phi[all_idx], psi[all_idx]], axis=-1)

    # secondary structure one-hot
    ss_letters = md.compute_dssp(traj, simplified=True)[all_idx]
    res_idx = residue_backbone_idx(list(range(traj.n_atoms)), traj.topology)
    ss_oh = np.zeros((len(all_idx), traj.n_atoms, 3), dtype=np.float32)
    for i, row in enumerate(ss_letters):
        for j, res in enumerate(res_idx):
            letter = ss_letters[i, j] if j < row.shape[0] else 'C'
            hot = SS3.get(letter, SS3['C'])
            for idx in res:
                if idx >= 0:
                    ss_oh[i, idx] = hot

    # create sliding windows for forecasting
    Xc, Xrg, Xcm, Xto, Xss, Y_ss, Y = process_windows(
        coords, rg, cmap, tors, ss_oh, args.window)

    # save
    with open(args.out, 'wb') as f:
        pickle.dump((Xc, Xrg, Xcm, Xto, Xss, Y_ss, Y, res_idx), f)
    print(f"Forecasting features saved to {args.out}")

if __name__ == '__main__':
    main()