"""
A hybrid system that continually updates 
    ChromaDB as users "use" the RAG, and then build the FAISS index once a day
    ChromaDB for dynamic indexing as users feed in new documents, questions, and feedback.
        stores metadata
    FAISS index is really fast but is a static index

Does FAISS require a fixed lenght?
Anything other than pickle?

LLama-Index for local models
https://colab.research.google.com/drive/16QMQePkONNlDpgiltOi7oRQgmB8dU5fl?usp=sharing

Haystack
"""
import asyncio
import chromadb
import chromadb.config
import datetime
import haystack
import haystack.document_stores
import haystack_integrations.components.retrievers.chroma
import operator
import pathlib
import sentence_transformers
import uuid

class DynamicDB:
    """
    Add documents to a dynamic collection or collections
    Assumes collections are mutually exclusive

    Supports metadata, dynamic collections, and federated search.

    Relies on LLamaIndex to abstract all of the functionality
        Use SentenceTransformer wrapper for embedddings
        Use ChromaDB wrapper for vectorDB
    """
    def __init__(self, fname_db, fname_embed, fname_llm, list_collections):
        """
        Fname existence is the user responsibility

        https://github.com/chroma-core/chroma/blob/main/chromadb/utils/embedding_functions/sentence_transformer_embedding_function.py
        """
        path_db = pathlib.Path(fname_db).resolve()
        # settings = chromadb.config.Settings(anonymized_telemetry=False)
        # self.client = chromadb.PersistentClient(settings=settings, path=str(path_db))

        # # Load local LLM
        # path_llm = pathlib.Path(fname_llm).resolve()
        # self.llm = llama_index.llms.llama_cpp.LlamaCPP(
        #     model_path=str(path_llm),
        #     max_new_tokens=512,
        #     context_window=4096, # 8192
        #     verbose=False
        # )

        # Load local SentenceTransformer
        path_embed = pathlib.Path(fname_embed).resolve()
        self.model_embed = sentence_transformers.SentenceTransformer(str(path_embed))

        # Collect the vector storage interfaces
        self.document_stores = {}
        for string_collection in list_collections:
            index = haystack_integrations.document_stores.chroma.ChromaDocumentStore(
                collection_name=string_collection,
                persist_path=str(path_db), # same path, different collection
                embedding_function="SentenceTransformerEmbeddingFunction",
                # embedding_function_params
                model_name=str(path_embed),
                device="cpu"
            )
            self.document_stores[string_collection] = index

    @staticmethod
    def construct_id(prefix="DOC"):
        date = datetime.datetime.utcnow()  # or datetime.now() for local time
        string_date = date.strftime("%Y%m%d-%H%M%S")  # e.g. "20250625-031045"
        string_random = str(uuid.uuid4())[:8]        # short UUID chunk
        out = f"{prefix}-{string_date}-{string_random}"

        return out

    def insert(self, string_collection, list_json):
        """
        Document should have the following structure
        {
            "text": str,
            "metadata": dict or None,
            "id": id or None
        }
        """
        index = self.document_stores[string_collection]
        list_docs = []
        list_ids = []
        key_document = "text"
        key_metadata = "metadata"
        key_identifier = "id"
        for _json in list_json:
            document = _json[key_document]
            metadata = _json.get(key_metadata, {})
            identifier = _json.get(key_identifier, self.construct_id(string_collection))

            doc = haystack.Document(
                content=document,
                meta=metadata,
                id=identifier,
            )
            list_docs.append(doc)
            list_ids.append(identifier)
        index.write_documents(list_docs)
        print(f"Inserted {len(list_docs)} documents into '{string_collection}'.")

        return list_ids

    def query(self, string_collection, query_string, filters=None, count_results=5):
        """
        Query using given text

        Functionally equivalent to
            query_embedding = embed_model.get_query_embedding(query_string)
            results = retriever.query_with_embedding(query_embedding)
        """
        index = self.document_stores[string_collection]
        retriever = haystack_integrations.components.retrievers.chroma.ChromaQueryTextRetriever(
            document_store=index,
            filters=filters,
            top_k=count_results
        )
        out = retriever.run(query=query_string)

        return out["documents"]

    def query_via_embedding(self, string_collection, embedding, filters=None, count_results=5):
        """
        Query using a precomputed embedding vector.

        NOTE: Does not support batch queries like ChromaDB natively
        """
        index = self.document_stores[string_collection]
        retriever = haystack_integrations.components.retrievers.chroma.ChromaEmbeddingRetriever(
            document_store=index,
            filters=filters,
            top_k=count_results
        )
        out = retriever.run(query_embedding=embedding)

        return out["documents"]

    def query_federated(self, query_string, filters=None, count_results=5):
        """
        Search across all collections.
        """
        best_score = float("-inf") # higher score = more relevant
        best_document = None

        for string_collection in self.document_stores.keys():

            list_documents = self.query(string_collection, query_string, filters, count_results)

            # pipeline = Pipeline()
            # pipeline.add_node(component=retriever, name="retriever", inputs=["Query"])
            # pipeline.add_node(component=self.llm, name="llm", inputs=["retriever"])

            # Run the pipeline with the query string
            # result = pipeline.run(query=query_string)

            # Extract the response and score
            # response = result["llm"]["response"]
            # documents = result["retriever"]["documents"]
            # response_score = documents[0].score if documents else 0.0  # Default score is 0.0 if no docs

            for document in list_documents:
                score = document.score

                if best_score < score:
                    best_document = document
                    best_score = score
                    best_collection = string_collection
                    # best_response = response

        return best_document

    def query_federated_via_embedding(self, embedding, filters=None, count_results=5):
        """
        Search across all collections
        """
        best_score = float("-inf") # higher score = more relevant
        best_document = None

        for string_collection in self.document_stores.keys():

            list_documents = self.query_via_embedding(string_collection, embedding, filters, count_results)

            # pipeline = Pipeline()
            # pipeline.add_node(component=retriever, name="retriever", inputs=["Query"])
            # pipeline.add_node(component=self.llm, name="llm", inputs=["retriever"])

            # Run the pipeline with the query string
            # result = pipeline.run(query=query_string)

            # Extract the response and score
            # response = result["llm"]["response"]
            # documents = result["retriever"]["documents"]
            # response_score = documents[0].score if documents else 0.0  # Default score is 0.0 if no docs

            for document in list_documents:
                score = document.score

                if best_score < score:
                    best_document = document
                    best_score = score
                    best_collection = string_collection
                    # best_response = response

        return best_document

# if __name__ == "__main__":

#     fname_db = "data/db"
#     fname_llm = "models/mistral-7b-v0.1.Q4_K_M.gguf" # for debugging tinyllama-1.1b-chat-v1.0.Q2_K.gguf
#     fname_embed = "models/all-MiniLM-L6-v2"
#     list_collections = ["docs"]

#     prompt = "Q: What is the capital of France?\nA:"

#     instance_db = DynamicDB(fname_db=fname_db, fname_embed=fname_embed, fname_llm=fname_llm, list_collections=list_collections)
#     output = instance_db.model_embed.encode(prompt)
#     print(output)

#     response = instance_db.query_federated("What are the latest advancements in AI?")
#     print(response)