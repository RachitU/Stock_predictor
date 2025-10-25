from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
try:
    import spacy
    nlp = spacy.load('en_core_web_sm')
except Exception:
    nlp = None
from .config import db, db_available

analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    if not text:
        return 0.0
    return analyzer.polarity_scores(text).get('compound', 0.0)

def extract_entities(text):
    if not nlp:
        return []
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ in ['ORG','PERSON','GPE','PRODUCT']]

def process_document(doc):
    content = doc.get('content') or doc.get('text') or doc.get('title') or ''
    sentiment = analyze_sentiment(content)
    entities = extract_entities(content)
    result = {'source_id': str(doc.get('_id', '')), 'entities': entities, 'sentiment': sentiment}
    if db_available:
        db.get_collection('nlp_analysis').insert_one(result)
    return result
