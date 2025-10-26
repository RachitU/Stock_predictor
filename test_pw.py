# test_pw.py
import os
from getpass import getpass
from src.config import check_password, users_collection, db_available, MONGO_URI

# Ensure environment variables are loaded if config relies on them implicitly
from dotenv import load_dotenv
load_dotenv()

print("--- Password Check Utility ---")

if not MONGO_URI:
    print("❌ MONGO_URI not found in environment. Cannot connect to DB.")
elif not db_available:
    print("❌ Database connection failed. Check MONGO_URI and network.")
elif users_collection is None:
    print("❌ Users collection object is None. Check DB connection/config.")
else:
    test_username = input("Enter the username to test: ").strip()
    plain_password_attempt = getpass("Enter the password you are trying: ") # Hides input

    if not test_username or not plain_password_attempt:
        print("❌ Username and password cannot be empty.")
    else:
        try:
            print(f"Searching for user '{test_username}'...")
            user_doc = users_collection.find_one({"username": test_username})

            if user_doc and 'password' in user_doc:
                hashed_pw_from_db = user_doc['password']
                print(f"Checking entered password against stored hash for user: {test_username}")

                # Call the check_password function from your config
                if check_password(hashed_pw_from_db, plain_password_attempt):
                    print("✅ Password MATCHES!")
                else:
                    print("❌ Password DOES NOT match!")
            elif user_doc:
                 print(f"❌ User '{test_username}' found, but 'password' field is missing in the database document!")
            else:
                print(f"❌ User '{test_username}' not found in the database.")

        except Exception as e:
            print(f"❌ An error occurred during database check: {e}")