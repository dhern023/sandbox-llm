"""
Recommended Inference models for CPU
    Phi-2 - A small but powerful model from Microsoft, optimized for reasoning and language understanding.
    Mistral 7B - A high-performance model that outperforms larger models like Llama 2 13B.
    Gemma 2B - A compact model from Google DeepMind, designed for efficient text generation.

    https://llama-cpp-python.readthedocs.io/en/latest/api-reference

Recommended Embed models for CPU
    | Model Name             | Dim | Notes                                 |
    | ---------------------- | --- | ------------------------------------- |
    | `all-MiniLM-L6-v2`     | 384 | Tiny, fast, great for general RAG     |
    | `bge-small-en-v1.5`    | 384 | Optimized for retrieval, very compact |
    | `intfloat/e5-small-v2` | 384 | Optimized for query ↔ doc matching    |
"""
import llama_cpp
import sentence_transformers

class InferenceModel:
    """
    Class interface saves the tokenizer and model instances
        so they can be quickly reused.
    """
    
    def __init__(self, fname=None, n_ctx=8192):
        """
        Creates a 
            llama_cpp.Llama model
            Calls the model's built-in tokenizer

        TODO: See if we can replace the tokenizer with a partitioner
        TODO: Check the sauce for Llama to see the default context window and if it can be inferred
        """
        self.n_ctx = n_ctx
        self.model = llama_cpp.Llama(model_path=fname, n_ctx=self.n_ctx)

    def tokenize(self, text, **kwargs):
        """
        Calls the llama model's built-in tokenizer
        """
        return self.model.tokenize(text, **kwargs)

    def infer(self, input, **kwargs):
        """
        Uses llama.cpp model api

        Arguments like max_tokens, stop strings
        """
        response = self.model(input, **kwargs)
        
        return response

class EmbedModel:
    """
    Class interface saves the tokenizer and model instances
        so they can be quickly reused.
    """

    def __init__(self, fname):
        """
        Load embedding model via sentence-transformers
        """
        self.model = sentence_transformers.SentenceTransformer(fname)

    def embed(self, list_texts):
        """
        Generate embeddings for a list of texts.
        Returns a list of float vectors.
        """
        return self.model.encode(list_texts, convert_to_tensor=False).tolist()

# if __name__ == "__main__":

#     fname_inference = "models/tinyllama-1.1b-chat-v1.0.Q2_K.gguf"
#     fname_embed = "models/all-MiniLM-L6-v2"

#     prompt = "Q: What is the capital of France?\nA:"
#     instance_infer = InferenceModel(fname=fname_inference)
#     output = instance_infer.infer(prompt)
#     print(output["choices"][0]["text"])

#     instance_embed = EmbedModel(fname=fname_embed)
#     output = instance_embed.embed([prompt])
#     print(output)