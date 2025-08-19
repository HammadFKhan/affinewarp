#%% 
"""
Data wrangling to correctly run spike warp pipeline on spiking data.
Although the actual model fit is good based on the imported data, 
we want to invoke the sparse warping function on the spike raster. 
Easiest way to do this is to format as intended
"""
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

fname = r"Y:\Hammad\Ephys\SeqProject\61691Mouse3Sq_Only\Day10\Day10DLSRecording1_250717_164317\UCLA_chanmap_fixed\spikes_to_Warp.mat"
D = loadmat(fname,squeeze_me=True)
print(D.keys())

from affinewarp import SpikeData
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
# %%

BINSIZE = 10.0   # ms
NBINS = int((data.tmax - data.tmin) / BINSIZE)
MAXLAG = 0.1
binned = data.bin_spikes(NBINS)

# # Uncomment to z-score...
# binned -= binned.mean(axis=(1, 2), keepdims=True)
# binned /= binned.std(axis=(1, 2), keepdims=True)

# %%
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
plt.plot(model.loss_hist)
# Validated spike raster transforms
#aligned_data = heldout_transform(model, binned, data);

# %%
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
#%%
# Example neurons shown in the paper.
neuron_ids = [0, 2, 5, 6, 9, 18]

# Create figure
fig, axes = plt.subplots(6, 5, figsize=(14, 12), sharex=True, sharey=True)
scatter_kw = dict(s=2, c='k', lw=0, alpha=.8)
line_kw = dict(lw=2, alpha=.5)
trial_range = np.arange(len(binned))
    
# Plot unwarped data
for n, ax in zip(neuron_ids, axes[:, 0]):
    idx = data.neurons == n
    y, x = data.trials[idx], data.spiketimes[idx]
    ax.scatter(x, y, **scatter_kw)
    ax.plot(pull3, np.arange(pull3.size), '-r', lw=2, alpha=.5)
    ax.plot(pull1, np.arange(ipi.size), '-b', lw=2, alpha=.5)
    ax.plot(pull2, np.arange(pull2.size), '-g', lw=2, alpha=.5)

# Plot data aligned to pull2
pull2_data = align_pull2.transform(data)

for n, ax in zip(neuron_ids, axes[:, 1]):
    idx = pull2_data.neurons == n
    y, x = pull2_data.trials[idx], pull2_data.spiketimes[idx]
    ax.scatter(x, y, **scatter_kw)
    t2 = align_pull2.event_transform(trial_range, pull3 / tmax) * tmax
    ax.plot(t2, trial_range, '-r', **line_kw)
    t1 = align_pull2.event_transform(trial_range, pull1 / tmax) * tmax
    ax.plot(t1, trial_range, '-b', **line_kw)
    t3 = align_pull2.event_transform(trial_range, pull2 / tmax) * tmax
    ax.plot(t3, np.arange(t3.size), '-g', lw=2, alpha=.5)

# Plot data aligned to pull1
pull1_data = align_pull1.transform(data)

for n, ax in zip(neuron_ids, axes[:, 2]):
    idx = pull1_data.neurons == n
    y, x = pull1_data.trials[idx], pull1_data.spiketimes[idx]
    ax.scatter(x, y, **scatter_kw)
    t2 = align_pull1.event_transform(trial_range, pull3 / tmax) * tmax
    ax.plot(t2, trial_range, '-r', **line_kw)
    t1 = align_pull1.event_transform(trial_range, pull1 / tmax) * tmax
    ax.plot(t1, trial_range, '-b', **line_kw)
    t3 = align_pull1.event_transform(trial_range, pull2 / tmax) * tmax
    ax.plot(t3, np.arange(t3.size), '-g', lw=2, alpha=.5)

# Plot data aligned to both lever presses
for n, ax in zip(neuron_ids, axes[:, 3]):
    _d = d.select_neurons(n)
    y, x = _d.trials, _d.spiketimes
    ax.scatter(x, y, **scatter_kw)
    t2 = align_both.event_transform(trial_range, pull3 / tmax) * tmax
    ax.plot(t2, trial_range, '-r', **line_kw)
    t1 = align_both.event_transform(trial_range, pull1 / tmax) * tmax
    ax.plot(t1, trial_range, '-b', **line_kw)
    t3 = align_both.event_transform(trial_range, pull2 / tmax) * tmax
    ax.plot(t3, np.arange(t3.size), '-g', lw=2, alpha=.5)

# Plot data with trials sorted by first pull.
kk = align_pull1.argsort_warps()
sorted_data = data.reorder_trials(kk)

for n, ax in zip(neuron_ids, axes[:,4]):
    idx = sorted_data.neurons == n
    y, x = sorted_data.trials[idx], sorted_data.spiketimes[idx]
    ax.scatter(x, y, **scatter_kw)
    ax.plot(pull1, trial_range[kk], '-b', **line_kw)
    ax.plot(pull3, trial_range[kk],'-r', **line_kw)


# Format axes.
for ax in axes.ravel():
    ax.set_xlim(0, tmax)
    ax.set_ylim(0, data.n_trials)

axes[0, 0].set_title("align pull 3")
axes[0, 1].set_title("align pull 2")
axes[0, 2].set_title("align pull 1")
axes[0, 3].set_title("align pull 2 and 3")
axes[0, 4].set_title("sorted by pull 1")

for s, ax in enumerate(axes[:, 0]):
    ax.set_ylabel("cell {}, trials".format(s))

fig.tight_layout()
# %%
from affinewarp.visualization import rasters

lin_aligned_data = model.transform(data)

rasters(lin_aligned_data,subplots=(6, 6))

rasters(data,subplots=(6, 6))
# %%
# %%
# Create dictionaries with correct metrics
import os
from scipy.io import savemat
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
# %%
