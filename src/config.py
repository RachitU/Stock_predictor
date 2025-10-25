import os
from dotenv import load_dotenv
from pymongo import MongoClient, errors

load_dotenv()

MONGO_URI = os.getenv('MONGO_URI')
DB_NAME = os.getenv('DB_NAME', 'synapse_street')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

db = None
db_available = False
if MONGO_URI:
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        # trigger connection
        client.server_info()
        db = client[DB_NAME]
        db_available = True
    except Exception as e:
        print('Warning: Could not connect to MongoDB:', e)
        db = None
        db_available = False
else:
    print('MONGO_URI not set -- running without DB.')
