# predicting-peak-bloom

## Model Predictions and Results
If you are here to see the predictions and the model features (with corresponding feature importance) you can find it in this [notebook](notebooks/Predictions.ipynb).

## Installation and Usage

1. You will need `python3` (preferably `python3.8`) to run this code.
2. If you use Ubuntu/Linux you can run the below commands in the terminal to get started.
3. If you use another operating-system, you can use your preferred python IDE to set this project up with a virtual environment and install dependencies from the `requirements.txt` file.

*Note regarding meteorological data used:* The latest meteorological data is automatically extracted from the code. The weather forecast for March 1 - March 7, 2022 is prone to change and may slightly effect the predicted results. If you want to use the data we used, pass `read_meteo_from_disk=True` when calling `generate_prediction_file()` function.   

#### 1. Clone the repository:
If you use `https`: 
```bash
git clone https://github.com/ankitdhall/predicting-peak-bloom.git
cd predicting-peak-bloom
```

If you use `ssh`:
```bash
git clone git@github.com:ankitdhall/predicting-peak-bloom.git
cd predicting-peak-bloom
```

If you want to download a `.zip`:
Alternatively, you can download the zip from the github website.

#### 2. Install virtual environment for python3.
```bash
pip3 install virtualenv
```

#### 3. Create a virtual environment to install dependencies:
```
virtualenv my_venv
source my_venv/bin/activate
```

#### 4. Use the `requirements.txt` to install dependencies.
```bash
pip3 install -r requirements.txt
```

#### 5. Make predictions:
Run this script to make predictions for 2022-2031.
```
cd utils
python3 make_predictions.py
```
You will find the predictions in the `predictions` folder as a `predicted_bloom_doy.csv` file.

---

#### [Extra] Interactive Jupyter notebook demo:
When you are in the parent directory run the following:
```
cd notebook
jupyter-lab
```
Your browser should open with the `jupyter-lab` interface.
1. Click on the "Folder" icon on the top-left and select the "Predictions.ipynb".
2. Once the notebook is open you can run the cells ("Kernel" > "Restart Kernel and run all cells...") to see the plots and predictions.
3. The features and their importance to predict the DOY for each model. 
