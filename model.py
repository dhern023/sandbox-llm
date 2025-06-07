"""
Recommended Models for CPU
    Phi-2 - A small but powerful model from Microsoft, optimized for reasoning and language understanding.
    Mistral 7B - A high-performance model that outperforms larger models like Llama 2 13B.
    Gemma 2B - A compact model from Google DeepMind, designed for efficient text generation.

    https://llama-cpp-python.readthedocs.io/en/latest/api-reference
"""
import llama_cpp

class ModelInterface:
    """
    Class interface saves the tokenizer and model instances
        so they can be quickly reused.
    """
    
    def __init__(self, fname_model=None, n_ctx=8192):
        """
        Creates a 
            llama_cpp.Llama model
            Calls the model's built-in tokenizer

        TODO: See if we can replace the tokenizer with a partitioner
        TODO: Check the sauce for Llama to see the default context window and if it can be inferred
        """
        self.n_ctx = n_ctx
        self.model = llama_cpp.Llama(model_path=fname_model, n_ctx=self.n_ctx)

    def tokenize(self, text, **kwargs):
        """
        Calls the llama model's built-in tokenizer
        """
        return self.model.tokenize(text, **kwargs)

    def inference(self, input, **kwargs):
        """
        Uses llama.cpp model api

        Arguments like max_tokens, stop strings
        """
        response = self.model(input, **kwargs)
        
        return response

# if __name__ == "__main__":
#     import llama_cpp

#     fname_model = "models/tinyllama-1.1b-chat-v1.0.Q2_K.gguf"
#     prompt = "Q: What is the capital of France?\nA:"
#     instance = ModelInterface(fname_model=fname_model)
#     output = instance.inference(prompt)
#     print(output["choices"][0]["text"])