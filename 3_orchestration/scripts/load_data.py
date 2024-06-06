import requests
from io import BytesIO
from typing import List

import pandas as pd

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader

@data_loader
def ingest_files(**kwargs) -> pd.DataFrame:

    def read_dataframe(filename):
        if filename.endswith('.csv'):
            df = pd.read_parquet(filename)

            df.lpep_dropoff_datetime = pd.to_datetime(df.lpep_dropoff_datetime)
            df.lpep_pickup_datetime = pd.to_datetime(df.lpep_pickup_datetime)
        elif filename.endswith('.parquet'):
            df = pd.read_parquet(filename)

        df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
        df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

        df = df[(df.duration >= 1) & (df.duration <= 60)]

        categorical = ['PULocationID', 'DOLocationID']
        df[categorical] = df[categorical].astype(str)
        
        return df


    df_train = read_dataframe('mlops/data/yellow_tripdata_2023-01.parquet')
    df_val = read_dataframe('mlops/data/yellow_tripdata_2023-02.parquet')

    return df_train, df_val