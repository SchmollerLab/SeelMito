import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import scipy.stats
import matplotlib

np.random.seed(3546843541)

matplotlib.rc('font', size=14)

this_path = os.path.abspath(__file__)
pwd_path = os.path.dirname(this_path)

def binned_mean_stats(x, y, nbins, bins_min_count):
    bin_counts, _, _ = scipy.stats.binned_statistic(x, y, statistic='count', bins=nbins)
    bin_means, bin_edges, _ = scipy.stats.binned_statistic(x, y, bins=nbins)
    bin_std, _, _ = scipy.stats.binned_statistic(x, y, statistic='std', bins=nbins)
    bin_width = (bin_edges[1] - bin_edges[0])
    bin_centers = bin_edges[1:] - bin_width/2
    x_errorbar = bin_centers[bin_counts>bins_min_count]
    y_errorbar = bin_means[bin_counts>bins_min_count]
    std = bin_std[bin_counts>bins_min_count]
    bin_counts = bin_counts[bin_counts>bins_min_count]
    std_err = std/np.sqrt(bin_counts)
    return x_errorbar, y_errorbar, std, std_err

csv_path = r"G:\My Drive\01_Postdoc_HMGU\Manuscripts\Anika_mito_2022\Anika_paper_Final_Data\spotMAX_Anika_WT_SCGE_Haploid_TOT_data.csv"

df = pd.read_csv(csv_path)

cell_vol_fl = df.cell_vol_fl
N = len(cell_vol_fl)

relative_err = 0.1

noise_cell_vol = np.random.normal(scale=relative_err, size=N)
cell_vol_predicted = cell_vol_fl + cell_vol_fl*noise_cell_vol

num_nucl = cell_vol_fl/2
noise_nucl = np.random.normal(scale=relative_err, size=N)
num_nucl += num_nucl*noise_nucl

mito_vol = cell_vol_fl*0.3
noise_mito = np.random.normal(scale=relative_err, size=N)
mito_vol += mito_vol*noise_mito

mito_conc = mito_vol/cell_vol_fl
num_nucl_conc = num_nucl/cell_vol_fl

mito_conc_pred = mito_vol/cell_vol_predicted
num_nucl_pred = num_nucl/cell_vol_predicted

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(mito_conc_pred, num_nucl_pred)

r_square = r_value**2
print(f'Predicted R square = {r_square:.3f}')

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(mito_conc, num_nucl_conc)

true_r_square = r_value**2
print(f'Predicted R square = {true_r_square:.3f}')

fig, ax = plt.subplots(1,3, figsize=(18,6))
fig.subplots_adjust(
    left=0.05, bottom=0.1, top=0.95, right=0.98
)

ax[0].scatter(mito_vol, num_nucl)
ax[1].scatter(mito_conc, num_nucl_conc)
ax[2].scatter(mito_conc_pred, num_nucl_pred)

ax[0].set_xlabel('Mito volume [a.u.]')
ax[0].set_ylabel('Number of nucleoids')

ax[1].set_xlabel('Mito volume/Cell volume [a.u./fL] TRUE VALUE')
ax[1].set_ylabel('Number of nucleoids/Cell volume [1/fL] TRUE VALUE')

nbins = 8
bins_min_count = 10
xe, ye, std, std_err = binned_mean_stats(mito_conc_pred, num_nucl_pred, nbins, bins_min_count)
ax[2].errorbar(xe, ye, yerr=std_err, c='k', capsize=3, lw=2)

ax[2].set_xlabel('Mito volume/Cell volume [a.u./fL] PREDICTED')
ax[2].set_ylabel('Number of nucleoids/Cell volume [1/fL] PREDICTED')

R_square_digits = str(r_square).split('.')[-1][:4]
true_R_square_digits = str(true_r_square).split('.')[-1][:4]
filename = f'modelling_relative_error_{int(relative_err*100)}perc_predicted_R_square_0-{R_square_digits}_true_R_square_0-{true_R_square_digits}.svg'
filepath = os.path.join(pwd_path, filename)
print(filename)
fig.savefig(filepath)

plt.show()