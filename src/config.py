import os
from dotenv import load_dotenv
from pymongo import MongoClient, errors

# Load environment variables from .env file
load_dotenv()

MONGO_URI = os.getenv('MONGO_URI')
DB_NAME = os.getenv('DB_NAME', 'synapse_street')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

db = None
db_available = False

# Increase timeout to 10 seconds (10000ms) to allow more time for network discovery.
CONNECTION_TIMEOUT_MS = 10000 

if MONGO_URI:
    try:
        # NOTE: Increased serverSelectionTimeoutMS from 5000 to 10000 for better resilience
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=CONNECTION_TIMEOUT_MS)
        # The .server_info() call attempts to connect and is where the error likely occurs
        client.server_info() 
        db = client[DB_NAME]
        db_available = True
        print('MongoDB connection successful.')
    except Exception as e:
        # This will now print the full error message in the console
        print('Warning: Could not connect to MongoDB:', e)
        db = None
        db_available = False
else:
    print('MONGO_URI not set -- running without DB.')
