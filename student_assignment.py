import datetime
import chromadb
import traceback
import pandas as pd

from chromadb.utils import embedding_functions

from model_configurations import get_model_configuration

gpt_emb_version = 'text-embedding-ada-002'
gpt_emb_config = get_model_configuration(gpt_emb_version)

dbpath = "./"

def init_collections():
    chroma_client = chromadb.PersistentClient(path=dbpath)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key = gpt_emb_config['api_key'],
        api_base = gpt_emb_config['api_base'],
        api_type = gpt_emb_config['openai_type'],
        api_version = gpt_emb_config['api_version'],
        deployment_id = gpt_emb_config['deployment_name']
    )
    collection = chroma_client.get_or_create_collection(
        name="TRAVEL",
        metadata={"hnsw:space": "cosine"},
        embedding_function=openai_ef
    )
    return collection

def load_csv():
    return pd.read_csv('COA_OpenData.csv')

def get_documents(df, tag):
    return df[tag].tolist()

def get_ids(df, tag):
    return df[tag].astype(str).tolist()


def generate_hw01():
    # init
    collection = init_collections()

    # load csv
    df = load_csv()

    # for basic
    documents = get_documents(df, 'HostWords')
    ids = get_ids(df, 'ID')

    # for metadata
    df['CreateDate'] = pd.to_datetime(df['CreateDate'])
    df['CreateDate'] = (df['CreateDate'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    metadata_columns = [col for col in df.columns if col not in ['ID', 'HostWords', 'FoodFeature']]
    metadatas = df[metadata_columns].to_dict('records')

    # add to collection
    collection.add(documents=documents, ids=ids, metadatas=metadatas)
    
    return collection

    
def generate_hw02(question, city, store_type, start_date, end_date):
    print(
    "question = " + str(question) + ",\n"
    "city = " + str(city) + ",\n"
    "store_type = " + str(store_type) + ",\n"
    "start_date = " + str(start_date) + ",\n"
    "end_date = " + str(end_date)
    )
    pass
    
def generate_hw03(question, store_name, new_store_name, city, store_type):
    print(
    "question = " + str(question) + ",\n"
    "store_name = " + str(store_name) + ",\n"
    "new_store_name = " + str(new_store_name) + ",\n"
    "city = " + str(city) + ",\n"
    "store_type = " + str(store_type)
    )
    pass
    
def demo(question):
    # init
    collection = init_collections()

    # load csv
    df = load_csv()

    # for basic
    documents = get_documents(df, 'HostWords')
    ids = get_ids(df, 'ID')

    # for metadata
    df['CreateDate'] = pd.to_datetime(df['CreateDate'])
    df['CreateDate'] = (df['CreateDate'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    metadata_columns = [col for col in df.columns if col not in ['ID', 'HostWords', 'FoodFeature']]
    metadatas = df[metadata_columns].to_dict('records')

    # add to collection
    collection.add(documents=documents, ids=ids, metadatas=metadatas)

    # test
    results = collection.query(query_texts=[question], n_results=2)
    print("查詢結果:")
    for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
        print(f"結果 {i+1}: 內容: {doc}, Metadata: {meta}")
    
    return collection


# demo("田媽媽")