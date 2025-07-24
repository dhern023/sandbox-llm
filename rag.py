"""
Nice to have:
    reranking the documents

Pipeline Overview
User Query
   ↓
Intent Classifier → ConditionalRouter
   ↓                     ↓
Query Expansion      Retrieval → Reranker
   ↓                     ↓
Clarification Prompt   Context Assembly
   ↓                     ↓
LLM Generation → ConditionalRouter
   ↓                     ↓
Answer or Fallback (e.g., document snippets or web search)

"""
DIR_DATA = pathlib.Path(__file__).parent / "data"
DIR_MODELS = pathlib.Path(__file__).parent / "models"
DIR_DOCS = DIR_DATA / "docs"
DIR_DB = DIR_DATA / "db" 

fname_embed = DIR_MODELS / "all-MiniLM-L6-v2"
fname_llm = DIR_MODELS / "tinyllama-1.1b-chat-v1.0.Q2_K.gguf" # for debugging mistral-7b-v0.1.Q4_K_M.gguf
list_collections = ["docs"]

dynamic_db = vector_db.DynamicDB(
        fname_db=DIR_DB,
        fname_embed=str(fname_embed), 
        fname_llm=str(fname_llm),
        list_collections=list_collections
    )

with open(DIR_DATA / "prompts" / "domains" / "legal.txt", 'r') as file:
    # Read the entire content of the file
    domain_instructions = file.read()
# EMBEDDING
text_embedder = haystack.components.embedders.SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")

# RETRIEVAL
index = dynamic_db.document_stores["docs"]
retriever = haystack_integrations.components.retrievers.chroma.ChromaEmbeddingRetriever(
    document_store=index,
    filters=None,
    top_k=5
)

# LLM
generator = haystack_integrations.components.generators.llama_cpp.LlamaCppChatGenerator(
    model=str(fname_llm),
    n_ctx=4096,
    n_batch=128,
    model_kwargs={"max_tokens": 512, "verbose": False},
)
generator.warm_up()

# PROMPTS 

dict_prompts = {
    # Prompt for domain-specific question
    "context": ChatPromptBuilder(
        [
            ChatMessage.from_system(domain_instructions),
            ChatMessage.from_system(
                """
                Answer the question using the provided context.
                {% if documents|length == 0 %}
                    Unable to answer directly.
                {% else %}
                    Use this context to answer the question:
                    {% for doc in documents %}
                        {{ doc.content }}
                    {% endfor %}
                {% endif %}
                """
            ),
            ChatMessage.from_user("Question: {{query}}"),
            ChatMessage.from_assistant("Answer: ")
        ],
        required_variables={"query", "documents"},
    ),
}

rag_pipeline = haystack.Pipeline()
# EMBEDDING
rag_pipeline.add_component("text_embedder", text_embedder)
# RETRIEVAL
rag_pipeline.add_component("retriever", retriever)
# LLM
rag_pipeline.add_component("generator", generator)
# PROMPTS
rag_pipeline.add_component("context", dict_prompts["context"])
# EMBEDDING
rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding") # embedding -> query_embedding parameter

# LLM
rag_pipeline.connect("retriever.documents", "context.documents") # send documents to prompt
rag_pipeline.connect("context", "generator.messages") # send documents to prompt

rag_pipeline.draw(path="rag_pipeline.png", super_component_expansion=True)

query = "@keyword: Roman architecture"

res = rag_pipeline.run(
    {
        "text_embedder": {"text": query},
        "context": {"query": query},
        # "router_queries": {"query": query},
    },
    include_outputs_from=["retriever"]
)

for d in res["retriever"]["documents"]:
    print(d.meta, d.score)

print(res)
