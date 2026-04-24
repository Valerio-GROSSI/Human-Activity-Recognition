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

2. Download the CSV files, with this command :
```bash
python Import_Data.py
```

4. Move the downloaded CSV files (Kaggle dataset) in the project folder. The downloaded CSV files path is displayed in the terminal.

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

Sample outputs on the MOT17 dataset :
