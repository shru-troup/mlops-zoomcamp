#!/usr/bin/env python
# coding: utf-8

import pickle
import pandas as pd
import argparse
import boto3

def read_data(filename):
    print(f'Reading file: {filename}')
    df = pd.read_parquet(filename)
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()    
    return df


def prepare_dictionaries(df: pd.DataFrame):
    categorical = ['PUlocationID', 'DOlocationID']
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    dicts = df[categorical].to_dict(orient='records')
    return dicts

def run():
    
    parser = argparse.ArgumentParser(description='Ride Duration Predictions...')
    parser.add_argument('--year', '-y', help="Taxi ride pick up year", type=int)
    parser.add_argument('--month', '-m', help="Taxi ride pick up month", type=int)
    
    args = parser.parse_args()
    year = args.year
    month = args.month

    input_filename = f'https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{year:04d}-{month:02d}.parquet'
    output_filename = f'data/prediction_{year:04d}-{month:02d}.parquet'
    
    
    df = read_data(filename=input_filename)
    dicts = prepare_dictionaries(df)
    
    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)
    
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred
    

    df_result.to_parquet(output_filename, engine='pyarrow', compression=None, index=False)
    
    s3 = boto3.client("s3")
    print(f'Uploading to S3 bucket..')
    s3.upload_file(Filename=output_filename, Bucket='mlflow-output', Key=output_filename)
    
    print(df_result.head())
    print(df_result['predicted_duration'].mean())

    
if __name__ == '__main__':
    run()
    


