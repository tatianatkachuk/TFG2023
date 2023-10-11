from pymongo import MongoClient

def get_db_handle(db_name, host, port, username, password):
   client = MongoClient(host=host,
                      port=int(port),
                      username=username,
                      password=password
                     )
   db_handle = client['met_data']
   return db_handle, client

