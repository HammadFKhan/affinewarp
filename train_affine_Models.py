#%% 
"""
Data wrangling to correctly run spike warp pipeline on spiking data.
Although the actual model fit is good based on the imported data, 
we want to invoke the sparse warping function on the spike raster. 
Easiest way to do this is to format as intended

Here we also introduce batch processing to quickly process all our spike data 
into the parent directory of where the data original came from. 
"""
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import glob
from affinewarp import SpikeData
from scipy.io import savemat
#%%
def grab_parent_file_loco(mat_file_path):
    with h5py.File(mat_file_path, 'r') as f:
        if 'ds_filename' not in f:
            print(f"Key 'ds_filename' not found in {mat_file_path}")
            return None
        dset = f['ds_filename']
        char_array = dset[:]
        if char_array.ndim > 1:
            char_array = char_array.flatten()
        if char_array.dtype == 'uint16':
            string_value = "".join([chr(c) for c in char_array])
        else:
            string_value = "".join([chr(c) for c in char_array])
    return string_value


def extract_warp_spikes(mat_file_path, save_path):
    print(f"Accessing: {mat_file_path}")
    parent_file = grab_parent_file_loco(mat_file_path)
    parent_directory, _ = os.path.split(parent_file)
    target_file = "spikes_to_warp.mat"
    fname = os.path.join(parent_directory, target_file)
    print(f"Accessing: {fname}")
    D = loadmat(fname,squeeze_me=True)
    print(D.keys())

    
    tmin = D["tmin"]
    tmax = D["tmax"]
    pull1 = D["pull1"]
    pull2 = D["pull2"]
    pull3 = D["pull3"]
    ipi = pull3 - pull1
    trials=D["trial_ids"]
    spiketimes=D["spiketimes"]
    neurons=D["neuron_ids"]
    data = SpikeData(
        trials=D["trial_ids"],
        spiketimes=D["spiketimes"],
        neurons=D["neuron_ids"],
        tmin=D["tmin"],
        tmax=D["tmax"],
    )

    BINSIZE = 10.0   # ms
    NBINS = int((data.tmax - data.tmin) / BINSIZE)
    MAXLAG = 0.1
    binned = data.bin_spikes(NBINS)

    # # Uncomment to z-score...
    # binned -= binned.mean(axis=(1, 2), keepdims=True)
    # binned /= binned.std(axis=(1, 2), keepdims=True)
    from affinewarp import PiecewiseWarping
    from affinewarp.crossval import heldout_transform

    # Create model.
    # Hyperparameters for shift-only warping model.
    SHIFT_SMOOTHNESS_REG = 0.02
    SHIFT_WARP_REG = 1e-2
    # Create model.
    model = PiecewiseWarping()
    #
    # Validated spike raster transforms
    #aligned_data = heldout_transform(model, DeltaFoverF)

    # Fit model to all neurons (for aligning behavior).
    model.fit(binned, iterations=50)
    print("Model fit complete")
    # Validated spike raster transforms
    #aligned_data = heldout_transform(model, binned, data);

    # Create manual warping, aligning to second lever press.
    t0 = np.column_stack((pull2 / tmax, np.full(pull2.size, np.median(pull2)/tmax)))
    align_pull2 = PiecewiseWarping(n_knots=0)
    align_pull2.manual_fit(binned, t0, recenter=True)

    # Create manual warping, aligning to first lever press.
    t0 = np.column_stack((pull1 / tmax, np.full(pull1.size, np.median(pull1)/tmax)))
    align_pull1 = PiecewiseWarping(n_knots=0)
    align_pull1.manual_fit(binned, t0, recenter=True)

    # Create manual warping, aligning to both lever press.
    align_both = PiecewiseWarping(n_knots=0)
    t0 = np.tile((pull3 / tmax)[:, None], (1, 2))
    t1 = np.column_stack((pull2 / tmax, np.full(ipi.size, np.median(pull2)/tmax)))
    align_both.manual_fit(binned, t0, t1, recenter=False)
    d = align_both.transform(data)


    scatter_kw = dict(s=2, c='k', lw=0, alpha=.8)
    line_kw = dict(lw=2, alpha=.5)
    trial_range = np.arange(len(binned))
        


    # Plot data aligned to pull2
    pull2_data = align_pull2.transform(data)

    pull1_data = align_pull1.transform(data)




    # Plot data with trials sorted by first pull.
    kk = align_pull1.argsort_warps()
    sorted_data = data.reorder_trials(kk)

    # Create dictionaries with correct metrics

    # Create a nested dictionary
    warpedSpikes = {
        'pull3A': {
            'pull1': pull1,
            'pull2': pull2,
            'pull3': pull3,
            'Spks': data
        },
        'pull2A': {
            'pull1': align_pull2.event_transform(trial_range, pull1 / tmax) * tmax,
            'pull2': align_pull2.event_transform(trial_range, pull2 / tmax) * tmax,
            'pull3': align_pull2.event_transform(trial_range, pull3 / tmax) * tmax,
            'Spks':  pull2_data
        },
        'pull1A': {
            'pull1': align_pull1.event_transform(trial_range, pull1 / tmax) * tmax,
            'pull2': align_pull1.event_transform(trial_range, pull2 / tmax) * tmax,
            'pull3': align_pull1.event_transform(trial_range, pull3 / tmax) * tmax,
            'Spks' : pull1_data
        },
        'pull23A': {
            'pull1': align_both.event_transform(trial_range, pull1 / tmax) * tmax,
            'pull2': align_both.event_transform(trial_range, pull2 / tmax) * tmax,
            'pull3': align_both.event_transform(trial_range, pull3 / tmax) * tmax,
            'Spks' : d
        }
    }


    directory, file_name = os.path.split(fname)
    file_name = "warpedSpks"
    # Construct the full path 
    file_path = os.path.join(directory, file_name + ".mat")
    # Save the data 
    savemat(file_path, {'warpedSpks': warpedSpikes})

    print(f"Model data saved: {os.path.exists(file_path)}")

def batch_train_affine_warp(directory_path):
    # Get all .mat files in the directory
    mat_files = glob.glob(os.path.join(directory_path, '*.mat'))

    # Create a folder for saving models
    models_dir = os.path.join(directory_path, 'affine_warp_models')
    os.makedirs(models_dir, exist_ok=True)

    
    print(f"Found {len(mat_files)} .mat files to process.")

    # Iterate through all mat files and train
    for mat_file in mat_files:
        # Building the output model file path by loading in the matfile
        # finding ds_loadme

        base_name = os.path.splitext(os.path.basename(mat_file))[0]
        save_path = os.path.join(models_dir, base_name)
        
        # Train the model from mat file
        extract_warp_spikes(mat_file, save_path)
        
    print(f'Trained and saved {len(mat_files)} models to {models_dir}')

# Here we pass the directory of all the spike files to quickly find out 
# where the original data resides. 

batch_train_affine_warp(r'Y:\Hammad\Ephys\SeqProject\ForceField')