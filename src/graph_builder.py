import networkx as nx
from .config import db, db_available

def build_sentiment_graph(entities, sentiment, topic='general'):
    G = nx.Graph()
    for e in entities:
        G.add_node(e, type='entity')
        G.add_node(topic, type='topic')
        G.add_edge(e, topic, weight=sentiment)
    if db_available:
        db.get_collection('graphs').insert_one({'entities': entities, 'sentiment': sentiment, 'topic': topic})
    return G
