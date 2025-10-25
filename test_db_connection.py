from src.config import db_available, db
print('DB available:', db_available)
if db_available:
    print('Collections:', db.list_collection_names())
