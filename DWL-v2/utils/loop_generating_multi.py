#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @author : Xujun Zhang, Tianyue Wang

'''
@File    :   loop_generating_data_perturb_ensemble.py (Data Perturbation - Minimal Output Change)
@Time    :   2023/03/09 19:31:32 (Modified for Data Perturbation Ensemble & MDN Selection)
@Author  :   Xujun Zhang, Tianyue Wang
@Version :   1.9
@Contact :   
@License :   
@Desc    :   Generates loop conformations using input data perturbation ensemble approach,
             selecting the best based on MDN score, aiming to keep original output format.
'''
import argparse
import os
import sys
import time
import torch
import torch.nn as nn
import pandas as pd
from prefetch_generator import BackgroundGenerator
from tqdm import tqdm
import numpy as np
import random
# import copy # Not strictly needed if PyG Batch.clone() is used

# dir of current
pwd_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(pwd_dir)
if os.path.exists(parent_dir):
    sys.path.append(parent_dir)
else:
    print(f"Warning: Parent directory {parent_dir} not found. Ensure utils, dataset, architecture are in PYTHONPATH.")

try:
    # Using the original set_random_seed from utils.fns
    from utils.fns import set_random_seed
    from utils.fns import save_loop_file
    from dataset.graph_obj import LoopGraphDataset
    from dataset.dataloader_obj import PassNoneDataLoader
    from architecture.Net_architecture import KarmaLoop # Ensure this is the version that evaluates pos_pred for MDN
except ImportError as e:
    print(f"Error importing custom modules: {e}")
    print("Please ensure that 'utils', 'dataset', and 'architecture' directories are in your PYTHONPATH or accessible.")
    sys.exit(1)


class DataLoaderX(PassNoneDataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class SingleSampleDataWrapper:
    """
    A wrapper to simulate the necessary parts of a PyG Batch object 
    for save_loop_file when processing a single selected sample.
    """
    def __init__(self, pdb_id_str, rdkit_mol, loop_indices_in_mol, 
                 predicted_coords_cpu, true_loop_coords_cpu=None):
        self.pdb_id = [pdb_id_str]
        self.mol = [rdkit_mol]
        self.loop_idx_2_mol_idx = [loop_indices_in_mol]
        self.pos_preds = predicted_coords_cpu

        class LoopStore:
            def __init__(self, num_atoms, true_pos=None):
                self.batch = torch.zeros(num_atoms, dtype=torch.long)
                if true_pos is not None:
                    self.pos = true_pos # Used if save_loop_file with out_init=True needs original loop pos
        
        num_atoms_in_loop = predicted_coords_cpu.shape[0]
        self._loop_store = LoopStore(num_atoms_in_loop, true_pos=true_loop_coords_cpu)

    def __getitem__(self, key): # Allows data['loop']
        if key == 'loop':
            return self._loop_store
        # Fallback for other attributes if needed, though save_loop_file primarily uses dot access for top level
        return getattr(self, key)


argparser = argparse.ArgumentParser()
argparser.add_argument('--graph_file_dir', type=str,
                       default='graph_100',
                       help='the graph files path')
argparser.add_argument('--model_file', type=str,
                       default='model_pkls/autoloop.pkl',
                       help='pretrained model file')
argparser.add_argument('--out_dir', type=str,
                       default='loop_result', 
                       help='dir for recording loop conformations and scores')
argparser.add_argument('--scoring', type=bool,
                       default=True,
                       help='whether predict loop generating scores (MUST be True for MDN selection)')
argparser.add_argument('--save_file', type=bool,
                       default=True,
                       help='whether save MDN-selected predicted loop conformations')
argparser.add_argument('--batch_size', type=int,
                       default=8,
                       help='batch size')
argparser.add_argument('--random_seed', type=int, # This will be the base seed
                       default=2020,
                       help='initial random_seed for perturbations and other stochasticity')
argparser.add_argument('--num_ensemble', type=int,
                       default=5, 
                       help='Number of ensemble runs with different data perturbations')
argparser.add_argument('--perturbation_scale', type=float,
                       default=0.1, # Angstroms
                       help='Standard deviation of Gaussian noise for coordinate perturbation')
args = argparser.parse_args()

if args.scoring is False:
    print("CRITICAL WARNING: --scoring is False, but selection is based on MDN score. This will likely lead to arbitrary selections or errors.")

print(f"Starting data perturbation ensemble generation with {args.num_ensemble} runs.")
print(f"Coordinate perturbation scale (std dev): {args.perturbation_scale} Angstroms.")

# Set initial global seed (from utils.fns)
set_random_seed(args.random_seed)

if not os.path.isdir(args.graph_file_dir):
    print(f"Error: graph_file_dir '{args.graph_file_dir}' not found or is not a directory.")
    sys.exit(1)

try:
    test_pdb_ids_files = [f for f in os.listdir(args.graph_file_dir) if os.path.isfile(os.path.join(args.graph_file_dir, f)) and not f.startswith('.')]
    if not test_pdb_ids_files:
        print(f"Error: No files found in graph_file_dir: {args.graph_file_dir}")
        sys.exit(1)
    test_pdb_ids = sorted(list(set(f.split('.')[0] for f in test_pdb_ids_files)))
    test_pdb_ids = sorted(test_pdb_ids, key=lambda x: int(x.split('_')[-1]) - int(x.split('_')[-2]) if len(x.split('_')) >= 4 and x.split('_')[-1].isdigit() and x.split('_')[-2].isdigit() else 0)
except Exception as e:
    print(f"Error processing pdb_ids from graph_file_dir: {e}")
    sys.exit(1)

test_dataset = LoopGraphDataset(src_dir='', # Assuming src_dir is not needed if graphs are pre-generated
                                dst_dir=args.graph_file_dir,
                                pdb_ids=test_pdb_ids,
                                dataset_type='test',
                                random_forward_reverse=False, # Typically False for testing
                                n_job=1, 
                                on_the_fly=True) # Assumes loading individual .dgl files

test_dataloader = DataLoaderX(dataset=test_dataset, batch_size=args.batch_size, shuffle=False,
                              num_workers=0, pin_memory=torch.cuda.is_available())

# device
device_id_str = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(',')[0]
if torch.cuda.is_available():
    try:
        device_id = int(device_id_str)
        torch.cuda.set_device(device_id)
        my_device = f'cuda:{device_id}'
    except ValueError:
        print(f"Warning: CUDA_VISIBLE_DEVICES ('{device_id_str}') is not a valid integer. Defaulting to cuda:0 or cpu.")
        my_device = 'cuda:0' if torch.cuda.device_count() > 0 else 'cpu'
else:
    my_device = 'cpu'
print(f"Using device: {my_device}")

model = KarmaLoop(hierarchical=True) 
model.to(my_device)

if not os.path.isfile(args.model_file):
    print(f"Error: Model file '{args.model_file}' not found.")
    sys.exit(1)

try:
    checkpoint = torch.load(args.model_file, map_location=my_device)
    if 'model_state_dict' in checkpoint: # Common way to save models
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    else: # If the file directly contains the state_dict
        model.load_state_dict(checkpoint, strict=True)
    print(f'# Model loaded from {args.model_file}')
except Exception as e:
    print(f"Error loading model weights: {e}")
    sys.exit(1)

overall_start_time = time.perf_counter()
best_predictions_info = {} 
os.makedirs(args.out_dir, exist_ok=True) # Main output directory
total_egnn_time_for_all_runs = 0 # Accumulates EGNN time across all runs

model.eval() # Set model to evaluation mode once

# Ensemble run loop
for ensemble_idx in range(args.num_ensemble):
    print(f"\n--- Ensemble Run {ensemble_idx + 1}/{args.num_ensemble} (Data Perturbation) ---")
    
    # Set random seed for this specific run to ensure different perturbations
    current_run_seed = args.random_seed + ensemble_idx # Simple way to vary seed
    set_random_seed(current_run_seed) 
    print(f"  Using random seed: {current_run_seed} for this run's perturbations.")
    
    current_run_total_egnn_time = 0 # EGNN time for this specific run
    with torch.no_grad():
        for batch_idx, data_batch_original in enumerate(tqdm(test_dataloader, desc=f"Ensemble {ensemble_idx+1}")):
            if data_batch_original is None: 
                print(f"Warning: DataLoader returned None for batch {batch_idx} in ensemble run {ensemble_idx+1}. Skipping.")
                continue
            
            # --- Data Perturbation ---
            data_batch_perturbed = data_batch_original.clone() # Deep copy for perturbation

            if args.perturbation_scale > 0:
                if 'protein' in data_batch_perturbed and hasattr(data_batch_perturbed['protein'], 'xyz'):
                    noise_xyz = torch.randn_like(data_batch_perturbed['protein'].xyz) * args.perturbation_scale
                    data_batch_perturbed['protein'].xyz = data_batch_perturbed['protein'].xyz + noise_xyz
                
                if 'protein' in data_batch_perturbed and hasattr(data_batch_perturbed['protein'], 'xyz_full'):
                    noise_xyz_full = torch.randn_like(data_batch_perturbed['protein'].xyz_full) * args.perturbation_scale
                    data_batch_perturbed['protein'].xyz_full = data_batch_perturbed['protein'].xyz_full + noise_xyz_full
            
            try:
                data_batch_for_model = data_batch_perturbed.to(my_device)
            except Exception as e:
                print(f"Error moving perturbed data to device in batch {batch_idx}, ensemble {ensemble_idx+1}: {e}")
                continue

            try:
                egnn_start_time_batch = time.perf_counter()
                # Assuming KarmaLoop.forward is modified to use its generated pos_pred for MDN scoring when train=False
                _, mdn_score_batch_tensor, pos_pred_batch_flat = model.forward(
                    data_batch_for_model, scoring=args.scoring, dist_threhold=5, train=False
                )
                current_run_total_egnn_time += time.perf_counter() - egnn_start_time_batch
                
                current_loop_atom_offset = 0
                for i in range(data_batch_for_model.num_graphs):
                    pdb_id = data_batch_for_model.pdb_id[i]
                    
                    loop_mask_for_sample_in_batch_atoms = (data_batch_for_model['loop'].batch == i)
                    if not torch.any(loop_mask_for_sample_in_batch_atoms):
                        continue

                    num_loop_atoms_sample = loop_mask_for_sample_in_batch_atoms.sum().item()
                    
                    # Use original (unperturbed) loop coordinates for RMSD calculation reference
                    pos_true_sample_original = data_batch_original['loop'].xyz[data_batch_original['loop'].batch == i].to(my_device)

                    pos_pred_sample = pos_pred_batch_flat[current_loop_atom_offset : current_loop_atom_offset + num_loop_atoms_sample]
                    current_loop_atom_offset += num_loop_atoms_sample

                    rmsd_current_sample = model.cal_rmsd(
                        pos_true_sample_original, 
                        pos_pred_sample, 
                        torch.zeros(num_loop_atoms_sample, dtype=torch.long, device=my_device) # Dummy batch for single sample RMSD
                    ).item()

                    mdn_score_current_sample = -float('inf')
                    if args.scoring and mdn_score_batch_tensor is not None:
                        if mdn_score_batch_tensor.numel() == data_batch_for_model.num_graphs:
                            mdn_score_current_sample = mdn_score_batch_tensor[i].item()
                        elif mdn_score_batch_tensor.numel() == 1: # If model returns a single score for the batch
                            mdn_score_current_sample = mdn_score_batch_tensor.item()
                    
                    current_best_mdn_for_pdb = best_predictions_info.get(pdb_id, {}).get('best_mdn_score', -float('inf'))
                    if mdn_score_current_sample > current_best_mdn_for_pdb:
                        best_predictions_info[pdb_id] = {
                            'best_mdn_score': mdn_score_current_sample,
                            'rmsd_at_best_mdn': rmsd_current_sample, 
                            'pos_pred_at_best_mdn': pos_pred_sample.clone().cpu(),
                            'original_mol': data_batch_original.mol[i], 
                            'loop_idx_in_mol': data_batch_original.loop_idx_2_mol_idx[i],
                            'pdb_id_str': pdb_id,
                            'loop_pos_true_for_save': pos_true_sample_original.clone().cpu(),
                        }
                    elif pdb_id not in best_predictions_info: # First time for this PDB
                        best_predictions_info[pdb_id] = {
                            'best_mdn_score': mdn_score_current_sample,
                            'rmsd_at_best_mdn': rmsd_current_sample,
                            'pos_pred_at_best_mdn': pos_pred_sample.clone().cpu(),
                            'original_mol': data_batch_original.mol[i],
                            'loop_idx_in_mol': data_batch_original.loop_idx_2_mol_idx[i],
                            'pdb_id_str': pdb_id,
                            'loop_pos_true_for_save': pos_true_sample_original.clone().cpu(),
                        }
            except Exception as e:
                print(f"Error processing batch {batch_idx} in ensemble {ensemble_idx+1}: {e}")
                continue
    total_egnn_time_for_all_runs += current_run_total_egnn_time
    print(f"--- Ensemble Run {ensemble_idx + 1} Finished. EGNN time for this run: {current_run_total_egnn_time/60:.2f} min ---")
    torch.cuda.empty_cache()

# Save PDB files for MDN-selected best conformations
# Path will be args.out_dir/0/{pdb_id}_pred.pdb to match original script's single run output structure
out_dir_for_selected_pdbs = os.path.join(args.out_dir, "0") # Create the "0" subdirectory
if args.save_file:
    print(f"\n--- Saving Best PDB Files to {out_dir_for_selected_pdbs} (selected by MDN score) ---")
    os.makedirs(out_dir_for_selected_pdbs, exist_ok=True)
    if not best_predictions_info:
        print("No best predictions were recorded to save.")
    else:
        for pdb_id, info in tqdm(best_predictions_info.items(), desc="Saving PDBs"):
            if info.get('pos_pred_at_best_mdn') is not None:
                save_data_obj = SingleSampleDataWrapper(
                    pdb_id_str=info['pdb_id_str'],
                    rdkit_mol=info['original_mol'],
                    loop_indices_in_mol=info['loop_idx_in_mol'],
                    predicted_coords_cpu=info['pos_pred_at_best_mdn'],
                    true_loop_coords_cpu=info.get('loop_pos_true_for_save')
                )
                try:
                    save_loop_file(save_data_obj, out_dir=out_dir_for_selected_pdbs, out_init=False, out_movie=False, out_pred=True)
                except Exception as e:
                    print(f"Error saving PDB for {pdb_id}: {e}")

# Prepare final statistics
final_rmsds_for_stats = []
final_mdn_scores_for_stats = []
final_pdb_ids_for_stats = []

if not best_predictions_info:
    print("No results collected from ensemble runs. Exiting statistics calculation.")
else:
    for pdb_id in test_pdb_ids: 
        if pdb_id in best_predictions_info:
            info = best_predictions_info[pdb_id]
            final_pdb_ids_for_stats.append(pdb_id)
            final_rmsds_for_stats.append(info['rmsd_at_best_mdn'])
            final_mdn_scores_for_stats.append(info['best_mdn_score'] if args.scoring else float('nan'))

# Save score.csv to args.out_dir (main directory)
if args.scoring:
    df_scores = pd.DataFrame({
        'pdb_id': final_pdb_ids_for_stats, 
        'confidence score': final_mdn_scores_for_stats
    })
    # Add RMSD to score.csv if needed, for easier correlation analysis later
    if final_rmsds_for_stats and len(final_rmsds_for_stats) == len(final_pdb_ids_for_stats):
        df_scores['rmsd_of_mdn_selected_vs_input_true'] = final_rmsds_for_stats
        
    df_scores.to_csv(os.path.join(args.out_dir, 'score.csv'), index=False)
    print(f"Scores for MDN-selected conformations saved to: {os.path.join(args.out_dir, 'score.csv')}")

# Calculate statistics for the final printout
data_statistic_single_run_equivalent = [] # This will hold one list of metrics, like original script's data_statistic[0]
if final_rmsds_for_stats:
    rmsds_tensor = torch.tensor(final_rmsds_for_stats, device=my_device if torch.cuda.is_available() else 'cpu')
    
    metrics_for_row = []
    metrics_for_row.append(rmsds_tensor.mean().item() if rmsds_tensor.numel() > 0 else float('nan'))
    metrics_for_row.append(rmsds_tensor.median().item() if rmsds_tensor.numel() > 0 else float('nan'))
    metrics_for_row.append(rmsds_tensor.max().item() if rmsds_tensor.numel() > 0 else float('nan'))
    metrics_for_row.append(rmsds_tensor.min().item() if rmsds_tensor.numel() > 0 else float('nan'))
    
    thresholds = [5, 4.5, 4, 3.5, 3, 2.5, 2, 1] # All original thresholds
    for t in thresholds:
        if rmsds_tensor.size(0) > 0:
            metrics_for_row.append((rmsds_tensor <= t).sum().float().item() / rmsds_tensor.size(0))
        else:
            metrics_for_row.append(0.0)
            
    # EGNN time: Original script reported egnn_time for the single 're' run.
    # Here, we report the total EGNN time for all ensemble runs.
    metrics_for_row.append(total_egnn_time_for_all_runs / 60) 
    data_statistic_single_run_equivalent.append(metrics_for_row)
else: # No RMSD stats to report
    data_statistic_single_run_equivalent.append([float('nan')] * 13)


# Mimic original script's mean and std calculation, even if data_statistic has only one entry
data_statistic_tensor = torch.as_tensor(data_statistic_single_run_equivalent)
data_statistic_mean = data_statistic_tensor.mean(dim=0) 
data_statistic_std = data_statistic_tensor.std(dim=0) # Will be 0 if only one "run" of stats

prediction_time_end = time.perf_counter()

# Final printout, matching original format
print(f'''
Total Time: {(prediction_time_end - overall_start_time) / 60:.2f} min
Sample Num: {len(test_dataset)}
# prediction (based on MDN-selected conformations from {args.num_ensemble} data-perturbed ensemble runs)
Time Spend: {data_statistic_mean[12]:.4f} ± {data_statistic_std[12]:.4f} min
Mean RMSD: {data_statistic_mean[0]:.4f} ± {data_statistic_std[0]:.4f}
Medium RMSD: {data_statistic_mean[1]:.4f} ± {data_statistic_std[1]:.4f}
Max RMSD: {data_statistic_mean[2]:.4f} ± {data_statistic_std[2]:.4f}
Min RMSD: {data_statistic_mean[3]:.4f} ± {data_statistic_std[3]:.4f}
Success RATE(5A): {data_statistic_mean[4]:.4f} ± {data_statistic_std[4]:.4f}
Success RATE(4.5A): {data_statistic_mean[5]:.4f} ± {data_statistic_std[5]:.4f}
Success RATE(4A): {data_statistic_mean[6]:.4f} ± {data_statistic_std[6]:.4f}
Success RATE(3.5A): {data_statistic_mean[7]:.4f} ± {data_statistic_std[7]:.4f}
Success RATE(3A): {data_statistic_mean[8]:.4f} ± {data_statistic_std[8]:.4f}
Success RATE(2.5A): {data_statistic_mean[9]:.4f} ± {data_statistic_std[9]:.4f}
Success RATE(2A): {data_statistic_mean[10]:.4f} ± {data_statistic_std[10]:.4f}
Success RATE(1A): {data_statistic_mean[11]:.4f} ± {data_statistic_std[11]:.4f}
''')

print(f"Data perturbation ensemble method completed. Best conformations selected from {args.num_ensemble} runs based on MDN scores.")
