# spotMAX
Python code used to analyse Seel et al. 2022

# System requirements
Windows 10 64 bit, macOS > 10

# Installation
## Typical installation time: 20 minutes
1. Install [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for **Python 3.9**.
*IMPORTANT: For Windows make sure to choose the **64 bit** version*.
2. Open a terminal and navigate to spotMAX-0.6 folder (this folder)
3. Update conda with `conda update conda`. Optionally, consider removing unused packages with the command `conda clean --all`
4. Update pip with `python -m pip install --upgrade pip`
5. Create a virtual environment with the command `conda create -n spotmax python=3.9`
6. Activate the environment with the command `conda activate spotmax`
7. Install all the dependencies with `pip install -r requirements.txt`

# Demo
## Expected run time: 10 minutes
1. Open a terminal and navigate to spotMAX-0.6/src folder
2. Run the command `python main_v1.py`
3. You will be prompted to select the folder containing the data. For this demo select the folder spotMAX-0.6/data/TIFFs
4. Follow instructions on the pop-up windows. Note that the pop-ups might be behind other open windows:
    4a. Select channel mNeon (mtDNA signal)
    4b. Select all 3 Positions
    4c. Answer yes to the cell cycle annotations pop-up
    4d. On the main parameters window click on "Load analysis inputs" and select the file spotMAX-0.6/data/6_v1_analysis_inputs.csv
    4c. As reference channel select mKate (mitochondria signal)
5. Wait for the analysis to end. At the end you get a window with 6 images. You can close that.
6. Results are saved in spotMAX-0.6/data/spotMAX_v1_run-num1/3_AllPos_p-_ellip_test_TOT_data.csv.

Relevant columns are "cell_vol_fl", "num_spots" (i.e., number of nucleoids), and "ref_ch_vol_um3" (i.e., mitochondria network volume).
If everything went fine you should get the same results that you can find in spotMAX-0.6/data/spotMAX_demo_results/3_AllPos_p-_ellip_test_TOT_data.csv.
