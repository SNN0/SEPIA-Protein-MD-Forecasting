# scripts/analyze.py

#!/usr/bin/env python3
"""
analyze.py - Compare predicted MD trajectories with original trajectories
This script is a refactored version of compare_traject_v2.py.
It loads a predicted trajectory and an original trajectory (both aligned
to the same starting frame) and compares structural properties. The script
generates plots and a summary report of key statistics.

- RMSD over time
- RMSF per atom
- Radius of gyration over time
- Secondary structure content (helix, sheet, coil) per frame
- Residue-level secondary structure evolution
- Contact maps for selected frames (with similarity measure excluding self­contacts)
- Backbone dihedral angle (phi/psi) comparison (averages, Ramachandran plots)
- Summary report of key statistics

Usage example:
  python analyze.py \
    --pred predicted_v13.dcd \
    --orig original_trajectory.dcd \
    --pdb structure.pdb \
    --start_idx 0 \
    --n_frames 150 \
    --out_dir analysis_results
"""
import os
import argparse
import numpy as np
import mdtraj as md
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, wasserstein_distance
import warnings
from scipy.stats import ConstantInputWarning

# Secondary structure color map (for plotting)
SS_COLORS = {
    'H': 'red',    # Alpha helix
    'E': 'yellow', # Beta sheet
    'C': 'grey'    # Coil
}

def load_data(args):
    """
    Load trajectories and prepare the slices for comparison.

    - Loads the PDB topology to select backbone atoms (N, CA, C, O).
    - Loads both predicted and original DCD trajectories with only these atoms.
    - Slices the original trajectory from `start_idx` for `n_frames` frames.
    - Truncates or warns if the predicted trajectory has fewer frames than requested.
    """
    print("Loading data...")

    topology = md.load_topology(args.pdb)
    sel_idxs = topology.select('name N or name CA or name C or name O')

    try:
        pred_traj = md.load(args.pred, top=args.pdb, atom_indices=sel_idxs)
    except Exception as e:
        raise RuntimeError(f"Failed to load predicted trajectory: {e}")

    try:
        orig_traj = md.load(args.orig, top=args.pdb, atom_indices=sel_idxs)
    except Exception as e:
        raise RuntimeError(f"Failed to load original trajectory: {e}")

    start_frame = args.start_idx
    end_frame = start_frame + args.n_frames
    if end_frame > orig_traj.n_frames:
        print(f"Warning: Requested end frame {end_frame} exceeds original "
              f"trajectory length {orig_traj.n_frames}. Using available frames.")
        end_frame = orig_traj.n_frames

    orig_slice = orig_traj[start_frame:end_frame]

    if pred_traj.n_frames < args.n_frames:
        print(f"Warning: Predicted trajectory has only {pred_traj.n_frames} frames, "
              f"but {args.n_frames} were requested")
        n_frames = min(pred_traj.n_frames, end_frame - start_frame)
    else:
        n_frames = end_frame - start_frame
        pred_traj = pred_traj[:n_frames]

    print(f"Using {n_frames} frames for comparison")
    return pred_traj, orig_slice

def calculate_rmsd(pred_traj, orig_traj):
    """
    Calculate RMSD over time between the predicted and original trajectories.

    For each frame, aligns the predicted frame to the corresponding original frame,
    then computes the RMSD (in nm). Returns an array of RMSD values.
    """
    print("Calculating RMSD...")
    n_frames = pred_traj.n_frames
    rmsd_values = np.zeros(n_frames)
    for i in range(n_frames):
        # Align this predicted frame to the i-th original frame
        aligned = pred_traj[i].superpose(orig_traj, frame=i)
        # Compute per-atom displacement, then the RMSD
        diff = aligned.xyz[0] - orig_traj.xyz[i]
        rmsd_values[i] = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
    return rmsd_values

def calculate_rmsf(pred_traj, orig_traj):
    """
    Calculate RMSF (root mean square fluctuation) per atom for each trajectory.

    Uses the average structure as the reference instead of the first frame.
    Returns two arrays: (predicted_rmsf, original_rmsf).
    """
    print("Calculating RMSF...")
    # Global alignment of predicted to original
    pred_aligned = pred_traj.superpose(orig_traj)

    # Build one‐frame average‐structure references
    pred_avg = pred_aligned[:1]
    pred_avg.xyz[0] = np.mean(pred_aligned.xyz, axis=0)
    orig_avg = orig_traj[:1]
    orig_avg.xyz[0] = np.mean(orig_traj.xyz, axis=0)

    # Compute RMSF relative to the average
    pred_rmsf = md.rmsf(pred_aligned, pred_avg)
    orig_rmsf = md.rmsf(orig_traj, orig_avg)
    return pred_rmsf, orig_rmsf

def calculate_radius_of_gyration(pred_traj, orig_traj):
    """
    Calculate radius of gyration (Rg) for each frame of the trajectories.
    Returns two arrays: (predicted_rg, original_rg).
    """
    print("Calculating Radius of Gyration...")
    pred_rg = md.compute_rg(pred_traj)
    orig_rg = md.compute_rg(orig_traj)
    return pred_rg, orig_rg

def calculate_secondary_structure(pred_traj, orig_traj):
    """
    Compute DSSP secondary structure per residue per frame for each trajectory,
    then count fractions of helix (H), sheet (E), and coil (C) per frame.
    """
    print("Calculating Secondary Structure content...")
    pred_ss = md.compute_dssp(pred_traj, simplified=True)
    orig_ss = md.compute_dssp(orig_traj, simplified=True)

    n_frames = pred_traj.n_frames
    n_res = pred_ss.shape[1]
    pred_ss_count = {ss: np.zeros(n_frames) for ss in ['H','E','C']}
    orig_ss_count = {ss: np.zeros(n_frames) for ss in ['H','E','C']}

    for i in range(n_frames):
        for ss in ['H','E','C']:
            pred_ss_count[ss][i] = np.mean(pred_ss[i] == ss)
            orig_ss_count[ss][i] = np.mean(orig_ss[i] == ss)

    return pred_ss, orig_ss, pred_ss_count, orig_ss_count


def calculate_contact_map_f1(pred_traj, orig_traj, frame_idx=0, cutoff=0.55):
    """
    Calculate binary contact maps and F1 score for a given frame.

    Contacts: atom-atom distance < cutoff (nm).
    Excludes diagonal (self-contacts) from comparisons.
    Returns:
      - pred_contacts: (N×N int array) predicted contact map
      - orig_contacts: (N×N int array) reference contact map
      - f1_score:      (float)   F1 score between the two maps
    """
    # 1) Extract coords at frame
    p = pred_traj[frame_idx].xyz[0]
    o = orig_traj[frame_idx].xyz[0]

    # 2) Compute distance matrices
    pred_dist = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
    orig_dist = np.linalg.norm(o[:, None, :] - o[None, :, :], axis=2)

    # 3) Binary contact maps
    pred_contacts = (pred_dist < cutoff)
    orig_contacts = (orig_dist < cutoff)

    # 4) Mask out diagonal
    N = pred_contacts.shape[0]
    mask = ~np.eye(N, dtype=bool)
    pc = pred_contacts[mask]
    oc = orig_contacts[mask]

    # 5) Compute TP, FP, FN
    TP = float(np.logical_and(pc, oc).sum())
    FP = float(np.logical_and(pc, ~oc).sum())
    FN = float(np.logical_and(~pc, oc).sum())

    # 6) Precision, Recall, F1
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1_score  = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return pred_contacts.astype(int), orig_contacts.astype(int), f1_score

def calculate_dihedral_angles(pred_traj, orig_traj):
    """
    Calculate backbone dihedral angles (phi/psi) for each residue over time.
    Returns arrays: pred_phi, orig_phi, pred_psi, orig_psi.
    """
    print("Calculating Dihedral Angles...")
    _, pred_phi = md.compute_phi(pred_traj)
    _, orig_phi = md.compute_phi(orig_traj)
    _, pred_psi = md.compute_psi(pred_traj)
    _, orig_psi = md.compute_psi(orig_traj)
    return pred_phi, orig_phi, pred_psi, pred_psi

def plot_rmsd(rmsd_values, out_dir):
    """Plot RMSD over time and save the figure."""
    plt.figure(figsize=(10, 6))
    plt.plot(rmsd_values, 'b-', linewidth=2)
    plt.title('RMSD between Predicted and Original Trajectories')
    plt.xlabel('Frame')
    plt.ylabel('RMSD (nm)')
    plt.grid(True, linestyle='--', alpha=0.7)
    # Add mean RMSD line
    mean_rmsd = np.mean(rmsd_values)
    plt.axhline(y=mean_rmsd, color='r', linestyle='--',
                label=f'Mean RMSD: {mean_rmsd:.4f} nm')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'rmsd_comparison.png'), dpi=300)
    plt.close()

def plot_rmsf(pred_rmsf, orig_rmsf, out_dir):
    """Plot RMSF for predicted vs original and include correlation and RMSE."""
    plt.figure(figsize=(12, 6))
    plt.plot(pred_rmsf, 'b-', label='Predicted', linewidth=2)
    plt.plot(orig_rmsf, 'r-', label='Original', linewidth=2)
    plt.title('RMSF Comparison')
    plt.xlabel('Atom Index')
    plt.ylabel('RMSF (nm)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    # Compute correlation and RMSE
    corr, p_value = pearsonr(pred_rmsf, orig_rmsf)
    rmse = np.sqrt(mean_squared_error(pred_rmsf, orig_rmsf))
    plt.figtext(0.15, 0.85, f'Corr: {corr:.4f} (p={p_value:.2e})\nRMSE: {rmse:.4f} nm')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'rmsf_comparison.png'), dpi=300)
    plt.close()

def plot_radius_of_gyration(pred_rg, orig_rg, out_dir):
    """Plot radius of gyration for predicted vs original, with statistics."""
    plt.figure(figsize=(10, 6))
    plt.plot(pred_rg, 'b-', label='Predicted', linewidth=2)
    plt.plot(orig_rg, 'r-', label='Original', linewidth=2)
    plt.title('Radius of Gyration Comparison')
    plt.xlabel('Frame')
    plt.ylabel('Rg (nm)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    # Correlation, RMSE, Wasserstein distance
    corr, p_value = pearsonr(pred_rg, orig_rg)
    rmse = np.sqrt(mean_squared_error(pred_rg, orig_rg))
    w_dist = wasserstein_distance(pred_rg, orig_rg)
    plt.figtext(0.15, 0.80,
                f'Corr: {corr:.4f} (p={p_value:.2e})\n'
                f'RMSE: {rmse:.4f} nm\n'
                f'Wasserstein: {w_dist:.4f}')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'rg_comparison.png'), dpi=300)
    plt.close()

def plot_secondary_structure(pred_ss_count, orig_ss_count, out_dir):
    """
    Plot the fraction of each secondary structure type over time for predicted vs original.

    Creates one subplot per type (H, E, C), including correlation and RMSE in the legend.
    """
    n_frames = len(pred_ss_count['H'])
    frames = np.arange(n_frames)
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    ss_names = {'H': 'α-helix', 'E': 'β-sheet', 'C': 'Coil'}
    for i, ss_type in enumerate(['H', 'E', 'C']):
        ax = axes[i]
        ax.plot(frames, pred_ss_count[ss_type], 'b-', label=f'Predicted {ss_type}', linewidth=2)
        ax.plot(frames, orig_ss_count[ss_type], 'r-', label=f'Original {ss_type}', linewidth=2)
        ax.set_title(f'{ss_names[ss_type]} Content Comparison')
        ax.set_ylabel('Fraction')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        # Compute statistics
        corr, p_value = pearsonr(pred_ss_count[ss_type], orig_ss_count[ss_type])
        rmse = np.sqrt(mean_squared_error(pred_ss_count[ss_type], orig_ss_count[ss_type]))
        ax.text(0.02, 0.85, f'Corr: {corr:.4f} (p={p_value:.2e})\nRMSE: {rmse:.4f}',
                transform=ax.transAxes)

    axes[-1].set_xlabel('Frame')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'secondary_structure_comparison.png'), dpi=300)
    plt.close()

def plot_residue_ss_evolution(pred_ss, orig_ss, out_dir):
    """
    Plot secondary structure (H/E/C) of each residue over time for both trajectories.

    Creates an image (residue vs. frame) for original and predicted, with a colorbar.
    Also prints overall secondary structure agreement.
    """
    n_frames, n_res = pred_ss.shape
    # Map SS letters to numeric values for plotting
    ss_map = {'H': 2, 'E': 1, 'C': 0}
    pred_num = np.zeros((n_frames, n_res))
    orig_num = np.zeros((n_frames, n_res))
    for i in range(n_frames):
        for j in range(n_res):
            pred_num[i, j] = ss_map.get(pred_ss[i, j], 0)
            orig_num[i, j] = ss_map.get(orig_ss[i, j], 0)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    cmap = plt.cm.get_cmap('viridis', 3)
    im1 = ax1.imshow(orig_num.T, aspect='auto', cmap=cmap, vmin=0, vmax=2, interpolation='none')
    ax1.set_title('Original Trajectory Secondary Structure')
    ax1.set_ylabel('Residue Index')
    im2 = ax2.imshow(pred_num.T, aspect='auto', cmap=cmap, vmin=0, vmax=2, interpolation='none')
    ax2.set_title('Predicted Trajectory Secondary Structure')
    ax2.set_ylabel('Residue Index')
    ax2.set_xlabel('Frame')
    # Colorbar for SS states
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im2, cax=cbar_ax)
    cbar.set_ticks([0,1,2])
    cbar.set_ticklabels(['Coil', 'β-sheet', 'α-helix'])
    # Calculate overall agreement fraction
    agreement = np.mean(pred_ss == orig_ss)
    plt.figtext(0.5, 0.01, f'Overall SS agreement: {agreement:.4f}', ha='center', fontsize=12)
    plt.tight_layout(rect=[0, 0.02, 0.9, 1])
    plt.savefig(os.path.join(out_dir, 'residue_ss_evolution.png'), dpi=300)
    plt.close()

def plot_contact_maps(pred_contacts, orig_contacts, similarity, out_dir, frame_idx=0):
    """
    Plot the contact maps of original, predicted, and their difference for a given frame.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    # Original contact map
    im1 = ax1.imshow(orig_contacts, cmap='viridis', origin='lower')
    ax1.set_title(f'Original Contact Map (Frame {frame_idx})')
    ax1.set_xlabel('Atom Index')
    ax1.set_ylabel('Atom Index')
    fig.colorbar(im1, ax=ax1)
    # Predicted contact map
    im2 = ax2.imshow(pred_contacts, cmap='viridis', origin='lower')
    ax2.set_title(f'Predicted Contact Map (Frame {frame_idx})')
    ax2.set_xlabel('Atom Index')
    ax2.set_ylabel('Atom Index')
    fig.colorbar(im2, ax=ax2)
    # Difference map
    diff = orig_contacts - pred_contacts
    im3 = ax3.imshow(diff, cmap='coolwarm', origin='lower', vmin=-1, vmax=1)
    ax3.set_title(f'Difference (F1-Score: {similarity:.4f})')
    ax3.set_xlabel('Atom Index')
    ax3.set_ylabel('Atom Index')
    fig.colorbar(im3, ax=ax3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'contact_maps_frame_{frame_idx}.png'), dpi=300)
    plt.close()

def plot_dihedral_comparison(pred_phi, orig_phi, pred_psi, orig_psi, out_dir):
    """
    Plot the average phi and psi angles by residue and example Ramachandran plots.

    - First, plot average phi and psi angle (per residue) for predicted vs original.
    - Then, for first/middle/last frames, plot phi vs psi scatter (Ramachandran).
    """
    avg_pred_phi = np.mean(pred_phi, axis=0)
    avg_orig_phi = np.mean(orig_phi, axis=0)
    avg_pred_psi = np.mean(pred_psi, axis=0)
    avg_orig_psi = np.mean(orig_psi, axis=0)

    # Plot average phi by residue
    plt.figure(figsize=(10, 6))
    plt.plot(avg_orig_phi, 'r-', label='Original', linewidth=2)
    plt.plot(avg_pred_phi, 'b-', label='Predicted', linewidth=2)
    plt.title('Average Phi Angles by Residue')
    plt.xlabel('Residue Index')
    plt.ylabel('Phi (radians)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    phi_corr, phi_p = pearsonr(avg_pred_phi, avg_orig_phi)
    phi_rmse = np.sqrt(mean_squared_error(avg_pred_phi, avg_orig_phi))
    plt.figtext(0.15, 0.85, f'Corr: {phi_corr:.4f} (p={phi_p:.2e})\nRMSE: {phi_rmse:.4f}')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'phi_angle_comparison.png'), dpi=300)
    plt.close()

    # Plot average psi by residue
    plt.figure(figsize=(10, 6))
    plt.plot(avg_orig_psi, 'r-', label='Original', linewidth=2)
    plt.plot(avg_pred_psi, 'b-', label='Predicted', linewidth=2)
    plt.title('Average Psi Angles by Residue')
    plt.xlabel('Residue Index')
    plt.ylabel('Psi (radians)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    psi_corr, psi_p = pearsonr(avg_pred_psi, avg_orig_psi)
    psi_rmse = np.sqrt(mean_squared_error(avg_pred_psi, avg_orig_psi))
    plt.figtext(0.15, 0.85, f'Corr: {psi_corr:.4f} (p={psi_p:.2e})\nRMSE: {psi_rmse:.4f}')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'psi_angle_comparison.png'), dpi=300)
    plt.close()

    # Ramachandran scatter for first, middle, last frames
    n_frames = pred_phi.shape[0]
    frames_to_plot = [0, n_frames//2, n_frames-1]
    for frame in frames_to_plot:
        plt.figure(figsize=(10, 8))
        plt.scatter(orig_phi[frame], orig_psi[frame], c='r', alpha=0.6, s=20, label='Original')
        plt.scatter(pred_phi[frame], pred_psi[frame], c='b', alpha=0.6, s=20, label='Predicted')
        plt.title(f'Ramachandran Plot (Frame {frame})')
        plt.xlabel('Phi (radians)')
        plt.ylabel('Psi (radians)')
        plt.xlim(-np.pi, np.pi)
        plt.ylim(-np.pi, np.pi)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'ramachandran_frame_{frame}.png'), dpi=300)
        plt.close()

def plot_ramachandran_density(pred_phi, orig_phi, pred_psi, orig_psi, out_dir):
    """
    Plot 2D density of phi/psi angles over the entire trajectory for original vs predicted.

    Also computes a 2D histogram-based Wasserstein distance between the distributions.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)
    # Flatten all angles
    orig_phi_flat = orig_phi.flatten()
    orig_psi_flat = orig_psi.flatten()
    pred_phi_flat = pred_phi.flatten()
    pred_psi_flat = pred_psi.flatten()
    # Kernel density plots
    sns.kdeplot(x=orig_phi_flat, y=orig_psi_flat, cmap="Reds", fill=True, ax=ax1, levels=10, thresh=0.05)
    sns.kdeplot(x=pred_phi_flat, y=pred_psi_flat, cmap="Blues", fill=True, ax=ax2, levels=10, thresh=0.05)
    # Titles and axes
    ax1.set_title('Ramachandran Density - Original')
    ax1.set_xlabel('Phi (radians)')
    ax2.set_title('Ramachandran Density - Predicted')
    ax2.set_xlabel('Phi (radians)')
    ax1.set_ylabel('Psi (radians)')
    for ax in [ax1, ax2]:
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-np.pi, np.pi)
        ax.grid(True, linestyle='--', alpha=0.5)
    # Discretize into histogram bins for Earth Mover's Distance
    bins = 36
    H_orig, xedges, yedges = np.histogram2d(orig_phi_flat, orig_psi_flat,
                                           bins=bins, range=[[-np.pi, np.pi], [-np.pi, np.pi]])
    H_pred, _, _ = np.histogram2d(pred_phi_flat, pred_psi_flat,
                                  bins=bins, range=[[-np.pi, np.pi], [-np.pi, np.pi]])
    # Normalize histograms
    H_orig = H_orig / np.sum(H_orig)
    H_pred = H_pred / np.sum(H_pred)
    # Compute 1D Wasserstein distance on the flattened distributions
    w_dist = wasserstein_distance(H_orig.flatten(), H_pred.flatten())
    plt.figtext(0.5, 0.01, f'Wasserstein Distance: {w_dist:.4f}',
                ha='center', fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(os.path.join(out_dir, 'ramachandran_density.png'), dpi=300)
    plt.close()

def create_summary_report(results, out_dir):
    """
    Create a text summary of key comparison metrics and save to a file.

    Metrics include RMSD statistics, RMSF/Rg/SS correlations & RMSE, contact similarity,
    dihedral angle correlations, and an overall qualitative assessment.
    """
    summary_file = os.path.join(out_dir, 'summary_metrics.txt')
    with open(summary_file, 'w', encoding="utf-8") as f:
        f.write("TRAJECTORY COMPARISON SUMMARY\n")
        f.write("=============================\n\n")
        # RMSD statistics
        f.write("RMSD Statistics:\n")
        f.write(f"  Mean RMSD: {np.mean(results['rmsd']):.4f} nm\n")
        f.write(f"  Min RMSD: {np.min(results['rmsd']):.4f} nm\n")
        f.write(f"  Max RMSD: {np.max(results['rmsd']):.4f} nm\n\n")
        # RMSF correlation & RMSE
        rmsf_corr, rmsf_p = pearsonr(results['pred_rmsf'], results['orig_rmsf'])
        f.write("RMSF Comparison:\n")
        f.write(f"  Pearson r: {rmsf_corr:.4f} (p={rmsf_p:.2e})\n")
        f.write(f"  RMSE: {np.sqrt(mean_squared_error(results['pred_rmsf'], results['orig_rmsf'])):.4f} nm\n\n")
        # Radius of gyration stats
        rg_corr, rg_p = pearsonr(results['pred_rg'], results['orig_rg'])
        f.write("Radius of Gyration:\n")
        f.write(f"  Pearson r: {rg_corr:.4f} (p={rg_p:.2e})\n")
        f.write(f"  RMSE: {np.sqrt(mean_squared_error(results['pred_rg'], results['orig_rg'])):.4f} nm\n")
        f.write(f"  Wasserstein dist: {wasserstein_distance(results['pred_rg'], results['orig_rg']):.4f}\n\n")
        # Secondary structure
        f.write("Secondary Structure Content:\n")
        for ss_type, ss_name in zip(['H','E','C'], ['α-helix','β-sheet','Coil']):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConstantInputWarning)
                ss_corr, ss_p = pearsonr(results['pred_ss_count'][ss_type], results['orig_ss_count'][ss_type])
            ss_rmse = np.sqrt(mean_squared_error(results['pred_ss_count'][ss_type], results['orig_ss_count'][ss_type]))
            f.write(f"  {ss_name}:\n")
            f.write(f"    Pearson r: {ss_corr:.4f} (p={ss_p:.2e})\n")
            f.write(f"    RMSE: {ss_rmse:.4f}\n")
            f.write(f"    Mean pred: {np.mean(results['pred_ss_count'][ss_type]):.4f}, Mean orig: {np.mean(results['orig_ss_count'][ss_type]):.4f}\n\n")
        # Contact map similarity
        f.write("Contact Map Similarity:\n")
        for frame, sim in results['contact_similarity'].items():
            f.write(f"  Frame {frame}: {sim:.4f}\n")
        f.write("\n")
        # Dihedral angles
        avg_pred_phi = np.mean(results['pred_phi'], axis=0)
        avg_orig_phi = np.mean(results['orig_phi'], axis=0)
        avg_pred_psi = np.mean(results['pred_psi'], axis=0)
        avg_orig_psi = np.mean(results['orig_psi'], axis=0)
        phi_corr, phi_p = pearsonr(avg_pred_phi, avg_orig_phi)
        psi_corr, psi_p = pearsonr(avg_pred_psi, avg_orig_psi)
        f.write("Dihedral Angles:\n")
        f.write(f"  Phi Corr: {phi_corr:.4f} (p={phi_p:.2e}), Phi RMSE: {np.sqrt(mean_squared_error(avg_pred_phi, avg_orig_phi)):.4f} rad\n")
        f.write(f"  Psi Corr: {psi_corr:.4f} (p={psi_p:.2e}), Psi RMSE: {np.sqrt(mean_squared_error(avg_pred_psi, avg_orig_psi)):.4f} rad\n\n")
        # Overall assessment
        avg_corr = np.mean([rmsf_corr, rg_corr, phi_corr, psi_corr,
                            pearsonr(results['pred_ss_count']['H'], results['orig_ss_count']['H'])[0],
                            pearsonr(results['pred_ss_count']['E'], results['orig_ss_count']['E'])[0],
                            pearsonr(results['pred_ss_count']['C'], results['orig_ss_count']['C'])[0]])
        f.write("OVERALL ASSESSMENT:\n")
        f.write(f"  Average correlation across metrics: {avg_corr:.4f}\n")
        mean_rmsd = np.mean(results['rmsd'])
        if mean_rmsd < 0.2:
            assessment = "EXCELLENT"
        elif mean_rmsd < 0.3:
            assessment = "GOOD"
        elif mean_rmsd < 0.5:
            assessment = "FAIR"
        else:
            assessment = "POOR"
        f.write(f"  Overall structural prediction quality: {assessment}\n")
    print(f"Summary report created: {summary_file}")

def parse_args():
    parser = argparse.ArgumentParser(description='Compare predicted and original trajectories')
    parser.add_argument('--features', type=str, required=False,
                        help='(Unused) Path to features pickle file')
    parser.add_argument('--pred', type=str, required=True,
                        help='Path to predicted trajectory file (DCD)')
    parser.add_argument('--orig', type=str, required=True,
                        help='Path to original trajectory file (DCD)')
    parser.add_argument('--pdb', type=str, required=True,
                        help='Path to PDB structure file for topology')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='Starting frame index in original trajectory')
    parser.add_argument('--n_frames', type=int, default=100,
                        help='Number of frames to compare')
    parser.add_argument('--out_dir', type=str, default='analysis_results',
                        help='Output directory for analysis results')
    return parser.parse_args()

def main():
    args = parse_args()
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
        print(f"Created output directory: {args.out_dir}")

    pred_traj, orig_traj = load_data(args)
    results = {}

    # RMSD
    results['rmsd'] = calculate_rmsd(pred_traj, orig_traj)
    plot_rmsd(results['rmsd'], args.out_dir)

    # RMSF
    results['pred_rmsf'], results['orig_rmsf'] = calculate_rmsf(pred_traj, orig_traj)
    plot_rmsf(results['pred_rmsf'], results['orig_rmsf'], args.out_dir)

    # Radius of gyration
    results['pred_rg'], results['orig_rg'] = calculate_radius_of_gyration(pred_traj, orig_traj)
    plot_radius_of_gyration(results['pred_rg'], results['orig_rg'], args.out_dir)

    # Secondary structure
    results['pred_ss'], results['orig_ss'], results['pred_ss_count'], results['orig_ss_count'] = \
        calculate_secondary_structure(pred_traj, orig_traj)
    plot_secondary_structure(results['pred_ss_count'], results['orig_ss_count'], args.out_dir)
    plot_residue_ss_evolution(results['pred_ss'], results['orig_ss'], args.out_dir)

    # Contact maps
    frames_to_analyze = [0, pred_traj.n_frames//2, pred_traj.n_frames-1]
    results['contact_similarity'] = {}
    for f in frames_to_analyze:
        pc, oc, sim = calculate_contact_map_f1(pred_traj, orig_traj, frame_idx=f)
        results['contact_similarity'][f] = sim
        plot_contact_maps(pc, oc, sim, args.out_dir, frame_idx=f)

    # Dihedral angles
    results['pred_phi'], results['orig_phi'], results['pred_psi'], results['orig_psi'] = \
        calculate_dihedral_angles(pred_traj, orig_traj)
    plot_dihedral_comparison(results['pred_phi'], results['orig_phi'],
                             results['pred_psi'], results['orig_psi'], args.out_dir)
    plot_ramachandran_density(results['pred_phi'], results['orig_phi'],
                              results['pred_psi'], results['orig_psi'], args.out_dir)

    # Summary
    create_summary_report(results, args.out_dir)
    print(f"Analysis complete. Results saved to {args.out_dir}")

if __name__ == "__main__":
    main()