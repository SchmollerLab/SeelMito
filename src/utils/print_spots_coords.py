import os
import sys

import pandas as pd
import tkinter as tk
from tkinter import ttk
import tkinter.filedialog

src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(src_path)

class single_combobox_widget:
    def __init__(self):
        self.is_first_call = True


    def prompt(self, values, title='Select value', message=None):
        root = tk.Toplevel()
        root.lift()
        # root.attributes("-topmost", True)
        root.title(title)
        root.geometry("+800+400")
        row = 0
        if message is not None:
            tk.Label(root,
                     text=message,
                     font=(None, 11)).grid(row=row, column=0,
                                           pady=(10,0), padx=10)
            row += 1

        # tk.Label(root,
        #          text='Select value:',
        #          font=(None, 11)).grid(row=row, column=0, pady=(10,0),
        #                                                 padx=10)
        w = max([len(v) for v in values])+10
        _var = tk.StringVar()
        _combob = ttk.Combobox(
            root, width=w, justify='center', textvariable=_var
        )
        _combob.option_add('*TCombobox*Listbox.Justify', 'center')
        _combob['values'] = values
        _combob.grid(column=0, row=row, padx=10, pady=(10,0))
        _combob.current(0)

        row += 1
        tk.Button(root, text='Ok', width=20,
                        command=self._close).grid(row=row, column=0,
                                                  pady=10, padx=10)



        root.protocol("WM_DELETE_WINDOW", self._abort)
        self._var = _var
        print(_var.get())
        self.root = root
        root.mainloop()

    def _close(self):
        self.selected_val = self._var.get()
        self.is_first_call = False
        self.root.quit()
        self.root.destroy()

    def _abort(self):
        self.root.quit()
        self.root.destroy()
        exit('Execution aborted by the user')

root = tk.Tk()

#expand dataframe beyond page width in the terminal
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 300)
pd.set_option('display.precision', 3)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

path = tkinter.filedialog.askopenfilename(master=root)

store = pd.HDFStore(path, mode='r')

df = store['frame_0']

store.close()

IDs = [str(ID) for ID in df.index.get_level_values(0).unique()]

win = single_combobox_widget()
win.prompt(
    IDs,
    title='Select cell ID',
    message='Select cell ID: '
)

ID = int(win.selected_val)

print(df.columns)

print(f'Spots for cell ID {ID}')
print(df.loc[ID, ['z', 'y', 'x', 'is_spot_inside_ref_ch']])
