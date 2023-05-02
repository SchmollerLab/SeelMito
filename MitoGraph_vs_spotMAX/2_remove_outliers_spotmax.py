import os

import pandas as pd

import utils

pwd_path = os.path.dirname(os.path.abspath(__file__))

spotmax_filtered_tables_path = os.path.join(pwd_path, 'spotmax_roi_filtered_tables')

# Outliers that Anika annotated as budding cells and I annotated as G1 cells
outliers = {
    'SCD_Diploid': [
        ('2020-02-18_Diploid_SCD_1', '2020-02-18_ASY15-1_150nM', 'Position_42', 1),
        ('2020-02-18_Diploid_SCD_1', '2020-02-18_ASY15-1_150nM', 'Position_49', 1),
        ('2020-02-18_Diploid_SCD_1', '2020-02-18_ASY15-1_150nM', 'Position_45', 3),
        ('2020-02-18_Diploid_SCD_1', '2020-02-18_ASY15-1_150nM', 'Position_28', 1),
        ('2020-02-18_Diploid_SCD_1', '2020-02-18_ASY15-1_150nM', 'Position_38', 2),
        ('2020-02-18_Diploid_SCD_1', '2020-02-18_ASY15-1_150nM', 'Position_16', 2),
        ('2020-02-18_Diploid_SCD_1', '2020-02-18_ASY15-1_150nM', 'Position_35', 6),
    ]
}

for f, file in enumerate(utils.listdir(spotmax_filtered_tables_path)):
    spotmax_final_df_path = os.path.join(spotmax_filtered_tables_path, file)
    df = pd.read_csv(spotmax_final_df_path).set_index([
        'replicate_num', 'horm_conc', 'Position_n', 'Moth_ID'
    ])
    for medium_haploid, _outliers in outliers.items():
        if file.find(medium_haploid) == -1:
            continue
        df = df.drop(index=_outliers)
        df.to_csv(spotmax_final_df_path)