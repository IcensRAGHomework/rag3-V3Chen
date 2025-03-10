import datetime
import chromadb
import traceback
import pandas as pd
import openai

from chromadb.utils import embedding_functions

from model_configurations import get_model_configuration

gpt_emb_version = 'text-embedding-ada-002'
gpt_emb_config = get_model_configuration(gpt_emb_version)

dbpath = "./"

# # set model
# openai_ef = embedding_functions.OpenAIEmbeddingFunction(
#     api_key="your_openai_api_key",
#     model_name="text-embedding-ada-002"
# )

# db initial
# client = chromadb.Client()
client = chromadb.PersistentClient(path=dbpath)
collection = client.get_or_create_collection(name="TRAVEL", metadata={"hnsw:space": "cosine"})
df = pd.read_csv('COA_OpenData.csv')

# for basic
documents = df['HostWords'].tolist()
ids = df['ID'].astype(str).tolist()

# for metadata
df['CreateDate'] = pd.to_datetime(df['CreateDate'])
df['CreateDate'] = (df['CreateDate'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
metadata_columns = [col for col in df.columns if col not in ['ID', 'HotWords', 'FoodFeature']]
metadatas = df[metadata_columns].to_dict('records')
               
# add to collection
collection.add(documents=documents, ids=ids, metadatas=metadatas)

# # test
# results = collection.query(query_texts=["田媽媽"], n_results=2)
# print("查詢結果:")
# for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
#     print(f"結果 {i+1}: 內容: {doc}, Metadata: {meta}")

def getCollection():
    return collection


def generate_hw01():
    return collection
    pass
    
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
