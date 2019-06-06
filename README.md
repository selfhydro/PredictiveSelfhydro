# PredictiveSelfhydro

## Running

Requires access to bigquery
Create a service account with access to GCP BigQuery and download json.

Run: `GOOGLE_APPLICATION_CREDENTIALS=./{file_name}.json python waterTemperaturePredictor.py`

#### Setup virtual env

1. `virtualenv --system-site-packages -p python3 ./venv`
1. `source ./venv/bin/activate`
1. `pip install --upgrade pip`
1. `pip install -r requirements.txt`
