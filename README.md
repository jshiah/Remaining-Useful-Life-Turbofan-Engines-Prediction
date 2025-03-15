#### Predicting the Remaining Useful Life of NASA Turbofan Engines 
DSC 288R, Project Group #: 8 
- Author(s): Joyce Shiah 
- Email: jshiah@ucsd.edu


Predicting the Remaining Useful Life (RUL) of turbofan engines is critical for ensuring aviation safety, reliability, and efficiency. This project introduces an adaptive sliding window approach that dynamically adjusts its size to capture real-time degradation trends under varying operating conditions. Unlike traditional fixed-window methods, this approach adapts to fluctuating degradation rates, improving prediction accuracy, human safety, and reducing unnecessary maintenance. By extending maintenance intervals for engines experiencing slower degradation and accelerating intervention for rapid deterioration, the method optimizes resource allocation and enhances equipment longevity. A refined sensor selection strategy – based on feature importance analysis – demonstrated improved performance compared to previous studies on the NASA C-MAPSS dataset. This approach contributes to improved safety, reduced maintenance costs, and increased environmental sustainability by minimizing premature component replacements.

Materials necessary for installation and set-up of Project reproduction:
1. `8-SW.zip` is the zip file for all source code and contains this `README.md` detailing instructions on how to compile and run the code. Download the .zip file and locate these key files: `train_FD001.txt`, `README.md`, `G8_Milestone4.ipynb`, `requirements.txt`, and `G8_Milestone4_Final_Report.pdf`.
2. To ensure a smooth experience with reproducing this project and its results, please run this project on your localhost Jupyter Notebook. If VS Code is strongly preferred, ensure your Python environment is set up correctly before proceeding any further. 
3. Take note of the `train_FD001.txt` facts before proceeding: The FD001 dataset (inside the .zip) is a simulated turbofan engine that operates on sea-level conditions with a high-pressure compressor degradation fault mode. FD001 contains 100 engine trajectories — each representing the operational history of a single engine unit — and a total of 20,631 performance snapshots. To use the dataset, download the file into your local drive and replace the Path Name with your local Path Name in this line: `df1 = pd.read_csv('/Users/JoyceShiah/Desktop/CMAPSSData/train_FD001.txt', sep="\s+", header=None, names=column_names, index_col=False)`. This line appears in the SECOND code chunk of the `G8_Milestone4.ipynb` Jupyter notebook.
Reference to dataset:
Saxena, A., & Goebel, K. (2008). Turbofan engine degradation simulation data set. NASA Prognostics Data Repository. NASA Ames Research Center.
4. Compile and ensure the first two code chunks, which contain the necessary imports of Python libraries, naming of columns, and the loading of the simulation dataset into the Pandas dataframe, all run successfully with no errors.
5. Once the first two code chunks are verified to have run successufully, please proceed to select from the Jupyter tabs "Run > Run All Cells." Please wait as the Notebook compiles to completion. As there are many algorithms that take up lots of computational resources and time, the estimated wait time for the successfull, complete compilation of `G8_Milestone4.ipynb` is: 7-8 minutes. There should be any compilation errors in this notebook, and only deprecation warnings will show up which you may ignore.
6. Please take the time to explore the contents of this script! All major decisions -- from data preprocessing, feature scaling, feature importance analysis, to model implementation -- include justification and discussions on key findings in Markdown cells. My entire thought process is relayed in these Markdown cells, and information parallel to the writing in the final report, `G8_Milestone4_Final_Report.pdf`, can be seen.
7. The step-by-step instructions as to how my pipeline went from exploratory data analysis, data preprocessing, feature scaling, feature importance analysis, to model implementation is fully flushed out in Markdown cells.

**Note:** The code, `G8_Milestone4.ipynb`, is already compiled on GitHub and models are all pre-trained. When downloading the .zip into your local system to reproduce the results, please use the GitHub version to compare with your findings.
Thank you very much in advance for reviewing my project! I hope you are able to gain some insight from my findings on accurate RUL prediction strategies.

Project status: Complete.


