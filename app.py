import os 
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
from mlflow.models.signature import infer_signature
import mlflow.sklearn
import logging
import random

logger = logging.getLogger(__name__)


def eval_metrics(actual,pred):
    rmse = np.sqrt(mean_squared_error(actual,pred))
    mae = mean_absolute_error(actual,pred)
    r2 = r2_score(actual,pred)

    return rmse,mae,r2


if __name__=="__main__":
    ## data ingestion

    csv_url = 'https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv'

    try:
        data = pd.read_csv(csv_url,sep = ';')
    except Exception as e:
        logger.exception(e)
    

    ## train test split


    y = data['quality']
    X = data.drop(columns = ['quality'],axis = 1)

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=random.randint(1,100))
    remote_server_uri = "http://ec2-13-221-14-181.compute-1.amazonaws.com:5000/"
    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    
    mlflow.set_tracking_uri(remote_server_uri)
    with mlflow.start_run():
        alpha= 1
        l1_ratio = 0.5
        lr = ElasticNet(alpha=alpha,l1_ratio=l1_ratio)
        lr.fit(X_train,y_train)

        predicted = lr.predict(X_test) 

        rmse,mae,r2 = eval_metrics(y_test,predicted)


        mlflow.log_param("alpha",alpha)
        mlflow.log_param("l1_ratio",l1_ratio)
        mlflow.log_metric('rmse',rmse)
        mlflow.log_metric('r2',r2)
        mlflow.log_metric('mae',mae)


        ## remote server aws setup
        
        


        if tracking_url_type_store!='file':
            mlflow.sklearn.log_model(
                lr,"model",registered_model_name="elasticnetWine"
            )
        else:
            mlflow.sklearn.log_model(lr,"model")
