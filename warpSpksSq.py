# %%
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm  # Import tqdm for loading bar
from scipy.io import savemat
from scipy.io import loadmat

# Import data from MATLAB built as behavior
# Collection of time series with shape (num_trials, num_timepoints,num_units).

fname = r"Y:\Hammad\Ephys\SeqProject\61691Mouse3Sq_Only\Day5\Day5DLSRecording1_250712_150101\UCLA_chanmap_fixed\spikes_to_Warp.mat"
mat = loadmat(fname,squeeze_me=True)
print(mat.keys())
DeltaFoverF = mat['SqSpikes']
leverPullIndex = mat['pullIndex']
print(f"Generated data shape: {DeltaFoverF.shape}")
tmax = np.size(DeltaFoverF,1)
tmin = 0
BINSIZE = 10  # ms
NBINS = int((tmax - tmin) / BINSIZE)
MAXLAG = 0.1
# Parameters
sigma = 1  # Smoothing parameter (in bins)

def compute_binned_spike_data(spike_counts, sigma, bin_size_ms,verbose=True):
    """
    Compute continuous firing rates from binned spike data using Gaussian smoothing.
    """

    # Check input dimensions
    if len(spike_counts.shape) != 2:
        raise ValueError(f"Expected 2D array, got shape {spike_counts.shape}")
    
    # 2. Bin the data into 20ms bins
    n_neurons = spike_counts.shape[1]
    n_timebins = spike_counts.shape[0]
    bin_size = bin_size_ms
    n_bins = n_timebins // bin_size
    binned_spike_data = np.zeros((n_bins, n_neurons))
    for i in range(n_bins):
        binned_spike_data[i] = spike_counts[i * bin_size:(i + 1) * bin_size].sum(axis=0)
    if verbose:
        print("Binned data shape:", binned_spike_data.shape)

    # Convert to Hz (spikes/second) by scaling
    scale_factor = bin_size_ms  # Convert to Hz
    smoothed_spike_data = np.zeros_like(binned_spike_data)
    # Apply Gaussian smoothing to each neuron individually
    for i in range(n_neurons):
        # Explicitly use array indexing
        current_neuron = binned_spike_data[:, i].copy()  # Get copy of this neuron's data
        # Scale first, then smooth
        smoothed_spike_data[:,i] = gaussian_filter1d(current_neuron * scale_factor, sigma=sigma)
    
    return smoothed_spike_data
spikes, time, trials = DeltaFoverF.shape
# Prepare binned output array: dimensions are (spikes, binned_time, trials)
binned_time = time // BINSIZE
binned_DeltaFoverF = np.zeros((spikes, binned_time, trials))
# Compute firing rates - make sure binned_spike_data is shape (neurons, time)
for n in range(trials):
    # For each trial, transpose to (time, spikes)
    trial_data = DeltaFoverF[:, :, n].T
    smoothed = compute_binned_spike_data(trial_data, sigma, BINSIZE,verbose=False)
    binned_DeltaFoverF[:, :, n] = smoothed.T  # Transpose back for correct axis orderted data shape: {binned_DeltaFoverF.shape}")
print(f"Generated data shape: {binned_DeltaFoverF.shape}")
DeltaFoverF = binned_DeltaFoverF
# %%
# # Uncomment to z-score...
#DeltaFoverF -= DeltaFoverF.mean(axis=(1, 2), keepdims=True)
#DeltaFoverF /= DeltaFoverF.std(axis=(1, 2), keepdims=True)

from affinewarp import ShiftWarping
from affinewarp import PiecewiseWarping
from affinewarp.crossval import heldout_transform
# Hyperparameters for shift-only warping model.
SHIFT_SMOOTHNESS_REG = 0.5
SHIFT_WARP_REG = 1e-2
# Create model.
model = PiecewiseWarping(n_knots=0, warp_reg_scale=SHIFT_WARP_REG, smoothness_reg_scale=SHIFT_SMOOTHNESS_REG,
                 l2_reg_scale=1e-7, min_temp=-3, max_temp=-1.5, n_restarts=3)
#
# Validated spike raster transforms
#aligned_data = heldout_transform(model, DeltaFoverF)

# Fit model to all neurons (for aligning behavior).
model.fit(DeltaFoverF, iterations=75)

plt.plot(model.loss_hist)
# %%
# 
# pred_spks is a 3 dimension array of trials, time, neurons
#pred_spks = model.predict()
kk = model.argsort_warps()

# Use a for loop to reorder the trials
trial_range = np.arange(len(kk))
aligned_data = model.transform(DeltaFoverF)
model_spks = model.predict()
# Create an empty array to hold the sorted data
sorted_aligned_data = np.empty_like(aligned_data)
for new_idx, old_idx in enumerate(kk):
    sorted_aligned_data[new_idx, :, :] = aligned_data[old_idx, :, :]

# Create time vector aligned to event at t=0
time_vector = np.linspace(-1.5, 3.5, sorted_aligned_data.shape[1])  # From -1.5s to 3.5s
# Create figure with subplots
# Plotting configuration

fig, axes = plt.subplots(5, 3, figsize=(15, 12))
axes = axes.flatten()

for neuron_idx in range(15):
    # Extract data for current neuron: trials × time
    neuron_data = DeltaFoverF[:, :, neuron_idx]
    
    # Create heatmap with time-aligned x-axis
    im = axes[neuron_idx].imshow(neuron_data,
                                aspect='auto',
                                cmap='viridis',
                                vmin=0,vmax=4,
                                extent=[time_vector[0], time_vector[-1], 0, neuron_data.shape[0]])
    
    # Add event marker at t=0
    axes[neuron_idx].axvline(x=0, color='red', linestyle='--', alpha=0.7)
    
    axes[neuron_idx].set_title(f'Neuron {neuron_idx+1}')
    axes[neuron_idx].set_xlabel('Time relative to movement (s)')
    axes[neuron_idx].set_ylabel('Trial #')
    fig.colorbar(im, ax=axes[neuron_idx], label='Spks')

plt.suptitle('Original Data', fontsize=16, y=0.99)
plt.tight_layout()

fig, axes = plt.subplots(5, 3, figsize=(15, 12))
axes = axes.flatten()

for neuron_idx in range(15):
    # Extract data for current neuron: trials × time
    neuron_data = sorted_aligned_data[:, :, neuron_idx]
    
    # Create heatmap with time-aligned x-axis
    im = axes[neuron_idx].imshow(neuron_data,
                                aspect='auto',
                                cmap='viridis',
                                vmin=0,vmax=4,
                                extent=[time_vector[0], time_vector[-1], 0, neuron_data.shape[0]])
    
    # Add event marker at t=0
    axes[neuron_idx].axvline(x=0, color='red', linestyle='--', alpha=0.7)
    
    axes[neuron_idx].set_title(f'Neuron {neuron_idx+1}')
    axes[neuron_idx].set_xlabel('Time relative to movement (s)')
    axes[neuron_idx].set_ylabel('Trial #')
    fig.colorbar(im, ax=axes[neuron_idx], label='Spks')

plt.suptitle('Warped and Aligned Data', fontsize=16, y=0.99)
plt.tight_layout()
plt.show()
# %%
# Calculate mean and standard error across trials
mean_response = np.mean(DeltaFoverF, axis=0)  # Shape: (150 time points, 10 neurons)
se_response = np.std(DeltaFoverF, axis=0) / np.sqrt(DeltaFoverF.shape[0])  # Standard error of the mean

# Create time vector aligned to event at t=0
time_vector = np.linspace(-1.5, 3.5, DeltaFoverF.shape[1])  # -1.5s to 3.5s

# Configure plot layout
fig, axes = plt.subplots(5, 3, figsize=(15, 12))
axes = axes.flatten()

for neuron_idx in range(15):
    # Plot mean response with shaded SE region
    axes[neuron_idx].plot(time_vector, mean_response[:, neuron_idx], 
                         color='navy', linewidth=2, label='Mean')
    
    axes[neuron_idx].fill_between(time_vector,
                                 mean_response[:, neuron_idx] - se_response[:, neuron_idx],
                                 mean_response[:, neuron_idx] + se_response[:, neuron_idx],
                                 color='skyblue', alpha=0.4, label='±1 SE')
    
    # Add event marker at t=0
    axes[neuron_idx].axvline(x=0, color='red', linestyle='--', alpha=0.7)
    
    # Formatting
    axes[neuron_idx].set_title(f'Neuron {neuron_idx+1}', fontsize=12)
    axes[neuron_idx].set_xlabel('Time relative to event (seconds)', fontsize=10)
    axes[neuron_idx].set_ylabel('Firing rate (a.u.)', fontsize=10)
    axes[neuron_idx].grid(alpha=0.2)

plt.suptitle('Original Neural Response', fontsize=16, y=0.99)
plt.tight_layout()
plt.show()

mean_response = np.mean(sorted_aligned_data, axis=0)  # Shape: (150 time points, 10 neurons)
se_response = np.std(sorted_aligned_data, axis=0) / np.sqrt(sorted_aligned_data.shape[0])  # Standard error of the mean

# Create time vector aligned to event at t=0
time_vector = np.linspace(-1.5, 3.5, sorted_aligned_data.shape[1])  # -1.5s to 3.5s

# Configure plot layout
fig, axes = plt.subplots(5, 3, figsize=(15, 12))
axes = axes.flatten()

for neuron_idx in range(15):
    # Plot mean response with shaded SE region
    axes[neuron_idx].plot(time_vector, mean_response[:, neuron_idx], 
                         color='navy', linewidth=2, label='Mean')
    
    axes[neuron_idx].fill_between(time_vector,
                                 mean_response[:, neuron_idx] - se_response[:, neuron_idx],
                                 mean_response[:, neuron_idx] + se_response[:, neuron_idx],
                                 color='skyblue', alpha=0.4, label='±1 SE')
    
    # Add event marker at t=0
    axes[neuron_idx].axvline(x=0, color='red', linestyle='--', alpha=0.7)
    
    # Formatting
    axes[neuron_idx].set_title(f'Neuron {neuron_idx+1}', fontsize=12)
    axes[neuron_idx].set_xlabel('Time relative to event (seconds)', fontsize=10)
    axes[neuron_idx].set_ylabel('Firing rate (a.u.)', fontsize=10)
    axes[neuron_idx].grid(alpha=0.2)

plt.suptitle('Aligned Neural Response', fontsize=16, y=0.99)
plt.tight_layout()
plt.show()
# %%
# Warp by lever pulls
pull1 = leverPullIndex[:,0]
pull2 = leverPullIndex[:,2]
ipi = pull2-pull1
# Create manual warping, aligning to both lever press.
align_both = PiecewiseWarping(n_knots=0)
t0 = np.column_stack((pull1 / tmax, np.full(pull1.size, .35)))
t0.shape[0] != DeltaFoverF.shape[0] 
#t1 = np.column_stack((pull2 / tmax, np.full(ipi.size, .5)))
align_both.manual_fit(DeltaFoverF, t0, recenter=False)
#%%
#t0 = np.tile((pull2/ tmax)[:, None], (1, 2))


kk = align_both.argsort_warps()

# Use a for loop to reorder the trials
trial_range = np.arange(len(kk))
aligned_lever_data = align_both.transform(DeltaFoverF)
model_lever_spks = align_both.predict()
# Create an empty array to hold the sorted data
sorted_aligned_lever_data = np.empty_like(aligned_lever_data)
for new_idx, old_idx in enumerate(kk):
    sorted_aligned_lever_data[new_idx, :, :] = aligned_lever_data[old_idx, :, :]

#pred_spks = align_both.predict()

# Create time vector aligned to event at t=0
time_vector = np.linspace(-1.5, 3.5, sorted_aligned_lever_data.shape[1])  # From -1.5s to 3.5s
# Calculate mean and standard error across trials
mean_response = np.mean(sorted_aligned_lever_data, axis=0)  # Shape: (150 time points, 10 neurons)
se_response = np.std(sorted_aligned_lever_data, axis=0) / np.sqrt(sorted_aligned_lever_data.shape[0])  # Standard error of the mean

# Create time vector aligned to event at t=0
time_vector = np.linspace(-1.5, 3.5, sorted_aligned_lever_data.shape[1])  # -1.5s to 3.5s

# Configure plot layout
fig, axes = plt.subplots(5, 4, figsize=(15, 12))
axes = axes.flatten()

for neuron_idx in range(20):
    # Plot mean response with shaded SE region
    axes[neuron_idx].plot(time_vector, mean_response[:, neuron_idx], 
                         color='navy', linewidth=2, label='Mean')
    
    axes[neuron_idx].fill_between(time_vector,
                                 mean_response[:, neuron_idx] - se_response[:, neuron_idx],
                                 mean_response[:, neuron_idx] + se_response[:, neuron_idx],
                                 color='skyblue', alpha=0.4, label='±1 SE')
    
    # Add event marker at t=0
    axes[neuron_idx].axvline(x=0, color='red', linestyle='--', alpha=0.7)
    
    # Formatting
    axes[neuron_idx].set_title(f'Neuron {neuron_idx+1}', fontsize=12)
    axes[neuron_idx].set_xlabel('Time relative to event (seconds)', fontsize=10)
    axes[neuron_idx].set_ylabel('Firing rate (a.u.)', fontsize=10)
    axes[neuron_idx].grid(alpha=0.2)

plt.suptitle('Average Neural Response', fontsize=16, y=0.99)
plt.tight_layout()
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.savefig("Warped_Lever_Neurons.pdf", format="pdf", bbox_inches="tight", transparent=True)
plt.show()
# %%
# Plot data with trials sorted by shift model.

# %%
# Create dictionaries with correct metrics
import os
warpedData = {
    'warpedSpks': aligned_data,
    'warpedSpks_sorted': sorted_aligned_data,
    'sortId':kk,
    'warpTemp': model_spks,
    'leverWarpedSpks': aligned_lever_data,
    'leverWarpedSpks_sorted': sorted_aligned_lever_data
}

directory, file_name = os.path.split(fname)
file_name = "warpedSpks"
# Construct the full path 
file_path = os.path.join(directory, file_name + ".mat")
# Save the data 
savemat(file_path, {'warpedSpks': warpedData})

print(f"Model data saved: {os.path.exists(file_path)}")
# %%
