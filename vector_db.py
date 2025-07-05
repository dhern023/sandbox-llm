"""
A hybrid system that continually updates 
    ChromaDB as users "use" the RAG, and then build the FAISS index once a day
    ChromaDB for dynamic indexing as users feed in new documents, questions, and feedback.
        stores metadata
    FAISS index is really fast but is a static index

Does FAISS require a fixed lenght?
Anything other than pickle?

"""
import chromadb
import chromadb.config
import pathlib
import uuid
import datetime

class DynamicDB:
    """
    Add documents to a dynamic collection or collections
    Assumes collections are mutually exclusive
    """
    def __init__(self, fname, list_collections):
        path = pathlib.Path(fname).resolve()
        settings = chromadb.config.Settings(anonymized_telemetry=False)
        self.client = chromadb.PersistentClient(settings=settings, path=str(path))
        self.collections = {
            name: self.client.get_or_create_collection(name) for name in list_collections
        }

    def get_collection(self, name):
        return self.collections.get(name)

    @staticmethod
    def construct_id(prefix="DOC"):
        date = datetime.datetime.utcnow()  # or datetime.now() for local time
        string_date = date.strftime("%Y%m%d-%H%M%S")  # e.g. "20250625-031045"
        string_random = str(uuid.uuid4())[:8]        # short UUID chunk
        out = f"{prefix}-{string_date}-{string_random}"

        return out

    def insert(self, string_collection, list_json, list_embeddings=None):
        """
        Document should have the following structure
        {
            "text": str,
            "metadata": dict or None,
            "id": id or None
        }
        """
        self.client.heartbeat() # ensure the client remains connected
        collection = self.get_collection(string_collection)
        # unwrap
        list_docs = []
        list_meta = []
        list_ids = []
        key_document = "text"
        key_metadata = "metadata"
        key_identifier = "id"
        for _json in list_json:
            document = _json[key_document]
            metadata = _json.get(key_metadata, {})
            identifier = _json.get(key_identifier, self.construct_id(string_collection))

            list_docs.append(document)
            list_meta.append(metadata)
            list_ids.append(identifier)
        
        # insert
        collection.add(documents=list_docs, metadatas=list_meta, ids=list_ids, embeddings=list_embeddings)
        print("Inserted into ChromaDB.")

        return list_ids

    def query(self, string_collection, list_embeddings, count_results):

        self.client.heartbeat() # ensure the client remains connected
        collection = self.get_collection(string_collection)
        # call the query
        dict_out = collection.query(
            query_embeddings=list_embeddings,
            n_results=count_results,
            include = ["embeddings", "metadatas", "documents", "distances"]
        )

        return dict_out

    def query_federated(self, list_embeddings, count_results):
        """
        Search across all collections
        """
        list_hits_all = []

        for collection_meta in self.client.list_collections():
            string_collection = collection_metadata.name

            list_hits = self.query(string_collection, list_embeddings, count_results)

            for i in range(len(list_hits["documents"][0])):
                list_hits_all.append({
                    "collection": name,
                    "doc": list_hits["documents"][0][i],
                    "metadata": list_hits["metadatas"][0][i],
                    "distance": list_hits["distances"][0][i]
                })

        list_out = sorted(list_hits_all, key=operator.itemgetter("distance"))

        return list_out

