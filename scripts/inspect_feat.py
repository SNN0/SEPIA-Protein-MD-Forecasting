
import argparse
import pickle
import numpy as np

SS_MAP = {
    (1,0,0): 'H (helix)',
    (0,1,0): 'E (strand)',
    (0,0,1): 'C (coil)'
}


def count_patterns(arr):
    # arr: (..., 3) array of 0/1 one‐hot
    flat = arr.reshape(-1, 3)
    pats, cnts = np.unique(flat, axis=0, return_counts=True)
    total = flat.shape[0]
    out = []
    for p, c in zip(pats, cnts):
        p_tup = tuple(p.tolist())
        label = SS_MAP.get(p_tup, str(p_tup))
        out.append((p_tup, label, c, 100*c/total))
    # sort by count descending
    return sorted(out, key=lambda x: -x[2])

def main():
    parser = argparse.ArgumentParser(description='Inspect feature pickle')
    parser.add_argument('--features', required=True, help='Path to features_mdtraj.pkl')
    parser.add_argument('--samples', type=int, default=3, help='Number of sample windows to inspect')
    args = parser.parse_args()

    # Load features
    with open(args.features, 'rb') as f:
        Xc, Xrg, Xcm, Xto, Xss, Xss_tgt, Y, res_idx = pickle.load(f)

    # Print top-level shapes
    print("Feature arrays:")
    print(f"  Xc       (coords):      {Xc.shape}  # (sample_frame, window, n_atom, 3)")
    print(f"  Xrg      (rg):          {Xrg.shape}  # (sample_frame, window)")
    print(f"  Xcm      (contact):     {Xcm.shape}  # (sample_frame, window, n_atom, n_atom)")
    print(f"  Xto      (torsions):    {Xto.shape}  # (sample_frame, window, n_res-1, 2)")
    print(f"  Xss      (ss one-hot):  {Xss.shape}  # (sample_frame, window, n_atoms, 3)")
    print(f"  Xss_tgt  (ss target):   {Xss_tgt.shape} # (sample_frame, n_atoms, 3)")
    print(f"  Y        (next coords): {Y.shape}  # (sample_frame, n_atoms, 3)")
    print(f"  res_idx  (mapping):     {len(res_idx)} residues")
    print("")

    # Data ranges
    print("Data ranges and types:")
    print(f"  coords: min={Xc.min():.4f}, max={Xc.max():.4f}")
    print(f"  rg:     min={Xrg.min():.4f}, max={Xrg.max():.4f}")
    print(f"  contact map unique values: {np.unique(Xcm)}")
    print(f"  torsions: min={Xto.min():.4f}, max={Xto.max():.4f} (radians)")
    print(f"  ss one-hot unique rows: {np.unique(Xss.reshape(-1,3), axis=0)}")
    print("")

    # Inspect samples
    n_windows = Xc.shape[0]
    m = min(args.samples, n_windows)
    for i in range(m):
        print(f"--- Sample_Frame {i+1}/{n_windows} ---")
        print(f"  RG window: {Xrg[i]}")
        cm0 = Xcm[i,0]
        print(f"  Contact map first frame shape: {cm0.shape}, total contacts: {cm0.sum()}")
        print(f"  First torsion (phi, psi) for residue 0: {Xto[i,0,0], Xto[i,0,1]}")
        ca_idx0 = res_idx[0][1]
        print(f"  First CA coords (nm) frame 0: {Xc[i,0, ca_idx0]}")
        print(f"  SS tgt residue 0 triple: {Xss_tgt[i,0, ca_idx0]}")
        print("")
        
    p = argparse.ArgumentParser()
    p.add_argument('--features', required=True,
                   help='Path to feature pickle (Xc, Xrg, Xcm, Xto, Xss, Xss_tgt, Y, res_idx)')
    args = p.parse_args()

    with open(args.features, 'rb') as f:
        Xc, Xrg, Xcm, Xto, Xss, Xss_tgt, Y, res_idx = pickle.load(f)

    print("\n=== Input windows SS patterns (Xss) ===")
    for pat, label, cnt, pct in count_patterns(Xss):
        print(f"  {label:>12s}  {pat} : {cnt} occurrences  ({pct:.2f}%)")

    print("\n=== Target SS patterns (Xss_tgt) ===")
    for pat, label, cnt, pct in count_patterns(Xss_tgt):
        print(f"  {label:>12s}  {pat} : {cnt} occurrences  ({pct:.2f}%)")

if __name__ == '__main__':
    main()