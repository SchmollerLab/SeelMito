import pandas as pd
import sys
import os

script_dir = os.path.dirname(os.path.realpath(__file__))
main_dir = os.path.dirname(script_dir)
sys.path.append(main_dir)

import prompts

#expand dataframe beyond page width in the terminal
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 300)
pd.set_option('display.precision', 3)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

path = prompts.file_dialog(title='Select .h5 file to convert to .csv')
filename, _ = os.path.splitext(os.path.basename(path))

if not path:
    sys.exit('Execution aborted by the user')

print('Reading file...')

store = pd.HDFStore(path, mode='r')

df = store['frame_0']

store.close()

print(df.head(10).iloc[:, :10])

path = prompts.folder_dialog(title='Select folder where to save .csv file')

if not path:
    sys.exit('Execution aborted by the user')

csv_path = os.path.join(path, f'{filename}.csv')

print('Saving file...')

df.to_csv(csv_path)

print('Done!')
