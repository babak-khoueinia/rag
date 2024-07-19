
import os

from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from langchain_fireworks import ChatFireworks

from tavily import TavilyClient

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core import StorageContext
from llama_index.core import Document, VectorStoreIndex
from llama_index.llms.fireworks import Fireworks
from llama_index.core import SimpleKeywordTableIndex, VectorStoreIndex
from langchain_fireworks import ChatFireworks
from llama_index.core import PromptTemplate as llama_prompt_template

tavily = TavilyClient(api_key="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")
os.environ["FIREWORKS_API_KEY"] = "7HO6KGKJOvVtfnG3kk7C64vAZYtAZWYAlx38ruWGke3pinfD"

Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
)

llm = ChatFireworks(model="accounts/fireworks/models/mixtral-8x7b-instruct")

def llama_Rag(question):
    
    response = tavily.search(query = question)
    
    context = [{"url": obj["url"], "content": obj["content"]} for obj in response['results']]
        
    documents = [Document(text=str(t)) for t in context]

    nodes = Settings.node_parser.get_nodes_from_documents(documents)

    # initialize storage context (by default it's in-memory)
    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes)

    vector_index = VectorStoreIndex(nodes, storage_context=storage_context)

    Settings.llm = llm
    
    query_engine = vector_index.as_query_engine(
        streaming = True,
        similarity_top_k = 10,
        retriever_mode="all_leaf",
        response_mode='tree_summarize',)
    
    new_summary_tmpl_str = (
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information and not prior knowledge, "
        "please answer the following question with citation to website link for each sentence that needs reffrence in the answer, and if the context was not enough to answer the quesion you must respond with FLAG and then give be best possible answer to the questions based on the context.\n"
        "Keep in mind to first add numeric reffrences and then list the websites that you used to answer the question.\n"
        "Query: {query_str}\n"
        "Answer: "
    )

    new_summary_tmpl = llama_prompt_template(new_summary_tmpl_str)
    query_engine.update_prompts({"response_synthesizer:summary_template": new_summary_tmpl})
    
    return query_engine

def rag_query(question, query_engine):
    
    response = query_engine.query(question)
    return response
