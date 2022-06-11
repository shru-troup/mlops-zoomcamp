import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


from prefect import flow, task, get_run_logger
from prefect.task_runners import SequentialTaskRunner

from datetime import datetime
from dateutil.relativedelta import relativedelta

import mlflow
import pickle


@task
def read_data(path):
    df = pd.read_parquet(path)
    return df

@task
def prepare_features(df, categorical, train=True):
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        print(f"The mean duration of training is {mean_duration}")
    else:
        print(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

@task
def train_model(df, categorical):
    #logger = prefect.context.get("logger")
    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    get_run_logger().info(f"The shape of X_train is {X_train.shape}")
    get_run_logger().info(f"The DictVectorizer has {len(dv.feature_names_)} features")
    # print(f"The shape of X_train is {X_train.shape}")
    # print(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    print(f"The MSE of training is: {mse}")
    return lr, dv

@task
def run_model(df, categorical, dv, lr):
    with mlflow.start_run():
        val_dicts = df[categorical].to_dict(orient='records')
        X_val = dv.transform(val_dicts) 
        y_pred = lr.predict(X_val)
        y_val = df.duration.values

        mse = mean_squared_error(y_val, y_pred, squared=False)
        # print(f"The MSE of validation is: {mse}")
        # mlflow.log_metric("rmse", mse)

         #mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")
         #mlflow.sklearn.log_model(lr, artifact_path="models_mlflow")
    return


@task
def get_paths(date):
    if date is None:
        today = datetime.today()
    else:
        today = datetime.strptime(date , '%Y-%m-%d')
    
    train_date = (today - relativedelta(months=2)).strftime('%Y-%m')
    val_date = (today - relativedelta(months=1)).strftime('%Y-%m')

    train_path = f'../data/fhv_tripdata_{train_date}.parquet'
    val_path = f'../data/fhv_tripdata_{val_date}.parquet'
    
    return train_path, val_path

@flow
def main(date=None):        
    
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("nyc-taxi-experiment-03")
    
    train_path, val_path = get_paths(date).result()
    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()
    run_model(df_val_processed, categorical, dv, lr)
    with open(f'dv_{date}.b', 'wb') as f_out:
        pickle.dump(dv, f_out)
    with open(f'model_{date}.bin', 'wb') as f_out:
        pickle.dump(lr, f_out)

from prefect import flow
from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule
from prefect.flow_runners import SubprocessFlowRunner


DeploymentSpec(
    name="cron-schedule-deployment",
    flow_location="homework.py",
    flow_runner=SubprocessFlowRunner(),
    schedule=CronSchedule(
        cron="0 9 15 * *",
        timezone="America/New_York"),
)
         
#main(date='2021-08-15')
