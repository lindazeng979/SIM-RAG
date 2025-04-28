#This is an EXAMPLE of how to use retriever with the hotpotqa wiki corpus.
import pickle
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search as BM25

# Load corpus and retriever settings
with open('bm25_search/corpus/corpus.pkl', 'rb') as f:
    corpus = pickle.load(f)
    #print(dict(list(corpus.items())[0:2]))
with open('bm25_search/corpus/retriever_settings.pkl', 'rb') as f:
    settings = pickle.load(f)

# Function to perform BM25 search on a new query
def bm25_search(query_text):
    index_name = settings['index_name']
    hostname = settings['hostname']
    number_of_shards = settings['number_of_shards']
    initialize = False  # Set to False as the index is already created
    
    model = BM25(index_name=index_name, hostname=hostname, initialize=initialize, number_of_shards=number_of_shards)
    retriever = EvaluateRetrieval(model)
    
    query = {"1": query_text}
    results = retriever.retrieve(corpus, query)
    query_id, scores_dict = list(results.items())[0]
    print(f"Query : {query[query_id]}")
    scores = sorted(scores_dict.items(), key=lambda item: item[1], reverse=True)
    for rank in range(10):
        doc_id = scores[rank][0]
        print(f"Doc {rank+1}: {doc_id} [{corpus[doc_id].get('title')}] - {corpus[doc_id].get('text')}")

# Example usage
bm25_search("What is the capital of France?")
