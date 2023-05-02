import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import scipy.stats
import matplotlib

matplotlib.rc('font', size=14)

this_path = os.path.abspath(__file__)
pwd_path = os.path.dirname(this_path)

tables_path = r"G:\My Drive\01_Postdoc_HMGU\Manuscripts\Anika_mito_2022\Anika_paper_Final_Data"
tables_names = ['SCD_Diploid', 'SCD_Haploid', 'SCGE_Diploid', 'SCGE_Haploid']


fig, ax = plt.subplots(2, 2, figsize=(16,16))
ax = ax.flatten()

for i, table_name in enumerate(tables_names):
    csv_filename = f'spotMAX_Anika_WT_{table_name}_TOT_data.csv'
    csv_filepath = os.path.join(tables_path, csv_filename)

    df = pd.read_csv(csv_filepath)
    mito_conc = df.ref_ch_vol_um3/df.cell_vol_fl
    num_nucl_conc = df.num_spots/df.cell_vol_fl

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(mito_conc, num_nucl_conc)

    r_square = r_value**2

    ax[i].scatter(mito_conc, num_nucl_conc)

    ax[i].set_xlabel('Mito volume/Cell volume [a.u./fL]')
    ax[i].set_ylabel('Number of nucleoids/Cell volume [1/fL]')
    ax[i].set_title(f'{table_name} - R square = {r_square:.3f}')

filename = f'spotmax_data_mito_conc_vs_num_nucl_conc.svg'
filepath = os.path.join(pwd_path, filename)
print(filename)
fig.savefig(filepath)

plt.show()