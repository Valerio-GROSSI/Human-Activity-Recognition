# Human Activity Recognition

This project focuses on Human Activity Recognition (HAR) using multivariate time-series data collected from smartphone IMU sensors.

It aims to classify different types of physical activities by training and evaluating supervised machine learning models on labeled datasets.

Please refer to the PDF in the current directory for more detailed information about this project and instructions on how to run it.
For the first run and setup, please refer to the instructions below.

<br>

## First Run

1. To have a python environment containing the expected libraries for running the scripts :
```bash
conda create -n name python=3.10
conda activate name
pip install -r requirements.txt
```

2. Download the CSV files (Kaggle dataset), with this command :
```bash
python Import_Data.py
```

4. Move the downloaded CSV files in the project folder. The path were the CSV files were downloaded is displayed in the terminal.

3. Execute the command :
```bash
python PyTorch_cv.py
```

5. Execute the following command in the results folder that was just created by the previous command :
```bash
cd logs/...
python analyse.py
```

<br>

## Sample Results

Sample outputs on the Kaggle dataset :

<p align="center">
<b>Optimal classifier using the 6 main variables</b><br>
Validation score: 96.8% ± 1.9%. Test score: 86.7%<br>
Confusion matrix on the test set :<br>
  <img src="logs/2025-09-01_19-14-30" width="80%">
</p>

<p align="center">
<b>Optimal classifier using the full feature set</b><br>
Validation score: 95.4% ± 2.3%. Test score: 95%<br>
Confusion matrix on the test set :<br>
<img src="logs/2025-09-03_18-21-58" width="80%">
</p>
