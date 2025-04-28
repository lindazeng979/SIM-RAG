"""
This example shows how to evaluate BM25 model (Elasticsearch) in BEIR on the HotpotQA dataset.
To be able to run Elasticsearch, you should have it installed locally (on your desktop) along with ``pip install beir``.
Depending on your OS, you would be able to find how to download Elasticsearch. I like this guide for Ubuntu 18.04 -
https://linuxize.com/post/how-to-install-elasticsearch-on-ubuntu-18-04/ 
For more details, please refer here - https://www.elastic.co/downloads/elasticsearch. 

This code doesn't require GPU to run.

BEIR Google Colab Demo, where they first install elastic search locally and retrieve using BM25
https://colab.research.google.com/drive/1HfutiEhHMJLXiWGT8pcipxT5L2TpYEdt?usp=sharing#scrollTo=nqotyXuIBPt6

"""

from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search as BM25

import pathlib, os, logging
import pickle

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

#### Download hotpotqa.zip dataset and unzip the dataset
dataset = "hotpotqa"
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)

#### Provide the data path where hotpotqa has been downloaded and unzipped to the data loader
# data folder would contain these files: 
# (1) hotpotqa/corpus.jsonl  (format: jsonlines)
# (2) hotpotqa/queries.jsonl (format: jsonlines)
# (3) hotpotqa/qrels/test.tsv (format: tsv ("\t"))

corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

#### Lexical Retrieval using Bm25 (Elasticsearch) ####
#### Provide a hostname (localhost) to connect to ES instance
#### Define a new index name or use an already existing one.
#### We use default ES settings for retrieval
#### https://www.elastic.co/

hostname = "localhost" #localhost
index_name = "hotpotqa" # hotpotqa

#### Initialize #### 
# (1) True - Delete existing index and re-index all documents from scratch 
# (2) False - Load existing index
initialize = True # False

#### Sharding ####
# (1) For datasets with small corpus (datasets ~ < 5k docs) => limit shards = 1 
# HotpotQA is a relatively small dataset! (limit shards to 1)
number_of_shards = 1
model = BM25(index_name=index_name, hostname=hostname, initialize=initialize, number_of_shards=number_of_shards)

# (2) For datasets with big corpus ==> keep default configuration
retriever = EvaluateRetrieval(model)

#### Save corpus and retriever settings for later use
with open('corpus.pkl', 'wb') as f:
    pickle.dump(corpus, f)
with open('retriever_settings.pkl', 'wb') as f:
    pickle.dump({
        'index_name': index_name,
        'hostname': hostname,
        'number_of_shards': number_of_shards
    }, f)

# Optional: Perform initial evaluation if needed
results = retriever.retrieve(corpus, queries)
logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)