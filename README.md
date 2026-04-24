# Human Activity Recognition

This project focuses on Human Activity Recognition (HAR) using multivariate time-series data collected from smartphone IMU sensors.

It aims to classify different types of physical activities by training and evaluating supervised machine learning models on labeled datasets.

Please refer to the PDF in the current directory for more detailed information about this project and instructions on how to run it.
For the first run and setup, please refer to the instructions below.

FIRST RUN:

1. To have a python environment containing the expected libraries for running the scripts :
conda create -n name python=3.10
conda activate name
pip install -r requirements.txt

2. Download the CSV files, with this command:
python Import_Data.py

3. Then move the downloaded CSV files in the project folder
The path were the CSV files were downloaded is displayed in the terminal

3. Execute the command
python PyTorch_cv.py

4. Execute the following command in the results folder that was just created by the previous command
cd logs/...
python analyse.py
