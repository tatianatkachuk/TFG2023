from pymongo import MongoClient
import pandas as pd

def returnDataframe(): 
    client = MongoClient()

    CONNECTION_STRING = "mongodb+srv://tatianatkachuk:VWGeJaHQDfnxx6YS@metdata.d4hd6wn.mongodb.net/"
    client = MongoClient(CONNECTION_STRING) 
    collection = client["met_data"]["dailydata"]

    data = collection.find({})

    print('Obtaining dataframe...')

    df = pd.DataFrame(data)

    print('Returning dataframe...')

    return df