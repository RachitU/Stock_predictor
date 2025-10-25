# SynapseStreet (Working Package)

Quick start:
1. unzip or clone this project
2. create and activate a virtualenv:
   python3 -m venv venv
   source venv/bin/activate
3. install deps:
   pip install --upgrade pip
   pip install -r requirements.txt
4. create a .env file (optional) or set environment variables:
   see .env.example
5. download spaCy model:
   python -m spacy download en_core_web_sm
6. run the Streamlit app:
   streamlit run app.py

Notes:
- The app uses MongoDB if MONGO_URI is set in .env, otherwise it runs in "no-db" mode.
- OpenAI features are optional; set OPENAI_API_KEY in .env if you want GPT summaries.
