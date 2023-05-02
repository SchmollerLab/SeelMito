# SeelMito
Python code used to analyse the microscopy data in [Seel et al. 2023](https://www.biorxiv.org/content/10.1101/2021.12.03.471050v2).

# System requirements
Windows 10 64 bit, macOS > 10

# Installation
## Typical installation time: 20 minutes
1. Install [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for **Python 3.9**.
*IMPORTANT: For Windows make sure to choose the **64 bit** version*.
2. Open a terminal and navigate to SeelMito folder (this folder)
3. Update conda with `conda update conda`. Optionally, consider removing unused packages with the command `conda clean --all`
4. Update pip with `python -m pip install --upgrade pip`
5. Create a virtual environment with the command `conda create -n seel python=3.9`
6. Activate the environment with the command `conda activate seel`
7. Install all the dependencies with `pip install -r requirements.txt`

# Demo
## Expected run time: 10 minutes
1. Open a terminal and navigate to SeelMito/src folder
2. Run the command `python main_v1.py`
3. You will be prompted to select the folder containing the data. For this demo select the folder SeelMito/data/TIFFs
4. Follow instructions on the pop-up windows. Note that the pop-ups might be behind other open windows:
    1. Select channel mNeon (mtDNA signal)
    2. Select all 3 Positions
    3. Answer yes to the cell cycle annotations pop-up
    4. On the main parameters window click on "Load analysis inputs" and select the file SeelMito/data/6_v1_analysis_inputs.csv
    5. As reference channel select mKate (mitochondria signal)
5. Wait for the analysis to end. At the end you get a window with 6 images. You can close that.
6. Results are saved in SeelMito/data/spotMAX_v1_run-num1/3_AllPos_p-_ellip_test_TOT_data.csv.

Relevant columns are "cell_vol_fl", "num_spots" (i.e., number of nucleoids), and "ref_ch_vol_um3" (i.e., mitochondria network volume).
If everything went fine you should get the same results that you can find in SeelMito/data/spotMAX_demo_results/3_AllPos_p-_ellip_test_TOT_data.csv.
