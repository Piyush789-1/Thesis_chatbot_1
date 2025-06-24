
import os

        
import streamlit as st 
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, models
from langchain_community.llms import LlamaCpp
from langchain.vectorstores import Chroma


@st.cache_resource
def load_backend_models():
    # All heavy initialization happens here:
    # --- Embedding model initialization ---
    local_model_path = "./models/sentence_transformer_all_mpnet_base_v2"
    transformer_model = models.Transformer(
        model_name_or_path=local_model_path,
        tokenizer_args={"local_files_only": True}
    )
    pooling_model = models.Pooling(transformer_model.get_word_embedding_dimension())
    sentence_transformer_model = SentenceTransformer(modules=[transformer_model, pooling_model])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sentence_transformer_model = sentence_transformer_model.to(device)
    print(f"Embedding model loaded on: {device}")

    # --- Vector store initialization ---
    class CustomSentenceTransformerEmbeddings:
        def __init__(self, model):
            self.model = model

        def embed_documents(self, texts):
            
            if isinstance(texts, str):
                texts = [texts]
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            
            
            if embeddings.ndim == 1:
                return [embeddings.tolist()]
            
            # If it's a 2D array, return the list of lists directly. no need to wrap it in list as output will be a list of lists.
            elif embeddings.ndim == 2:
                return embeddings.tolist()
            else:
                # Fallback, though typically not needed.
                return [emb.tolist() for emb in embeddings]

        def embed_query(self, text):
            embedding = self.model.encode(text, convert_to_numpy=True)
            # If the embedding comes back as a 2D array (e.g., shape [1, d]), get the first element.
        
            if isinstance(embedding, np.ndarray):
                if embedding.ndim == 2:
                    embedding = embedding[0]
                return embedding.tolist()
            return embedding
    
    persist_directory = r"D:\ML\Thesis_chatbot\Data\out\chroma_db"
    vectorstore = Chroma(
        persist_directory=persist_directory,
        collection_name="my_collection",
        embedding_function=CustomSentenceTransformerEmbeddings(sentence_transformer_model)
    )
    print("Vector store loaded.")

    # --- LLM initialization ---
    
    if torch.cuda.is_available():
        print("✅ GPU is available.")
    else:
        print("❌ GPU is not available.")
    
    llm = LlamaCpp(
        model_path=r"D:\ML\Thesis_chatbot\Models\TheBloke_OpenHermes-2.5-Mistral-7B-GGUF\openhermes-2.5-mistral-7b.Q4_K_M.gguf",
        n_gpu_layers=-1,
        n_ctx=32768,
        n_batch=1024,
        n_threads=8
    )
    print("LLM loaded.")

    
    return vectorstore, llm

# Load heavy resources once, on-demand
vectorstore, llm = load_backend_models()


def answer_query(query: str) -> str:
    """
    Generates an answer for the given query using both a similarity search for document context
    and recent conversation history (if available via Streamlit's session_state).

    The prompt is constructed by concatenating:
      - Recent conversation turns (if any)
      - Retrieved document context (via the vectorstore)
      - The current query

    Finally, the prompt is passed to the LLM to generate a response.
    """
    print("Inside answer_query with conversation history integration")
    
    # Retrieve the document context from your vector store.
    body_docs = vectorstore.similarity_search(query=query, k=3)
    if not body_docs:
        return "No documents were retrieved. Check your vector store or query settings."

    context_chunks = []
    for doc in body_docs:
        chunk = ""
        if hasattr(doc, "metadata") and doc.metadata:
            chunk += f"Metadata: {doc.metadata}\n"
        chunk += f"Content:\n{doc.page_content}"
        context_chunks.append(chunk)
    document_context = "\n\n".join(context_chunks)
    
    
    # Retrieve conversation history from Streamlit's session_state (if available)
    try:
        import streamlit as st
        conversation_history = st.session_state.get("conversation_history", [])
    except Exception as e:
        conversation_history = []
    
    # Prepare a conversation context from the last few turns (you can adjust the number as needed)
    conversation_context = ""
    for message in conversation_history[-4:]:
        # Format each turn: "User:" or "Bot:" followed by the message.
        role_label = "User" if message["role"] == "user" else "Bot"
        conversation_context += f"{role_label}: {message['message']}\n"
    
    # Construct the prompt including conversation history, document context, and current query.
    prompt = (
        "Instruction:\n"
        "- Provide a concise, precise final answer.\n"
        "- Do not hallucinate.\n"
        "- The tables in retrieved chunks are in markdown format.\n"
        "- Check metadata to identify relevant chunks if required.\n"
        "- If your answer involves an image, include it using the markdown format: ![IMG_TITLE: Your Caption](your/image/path.jpeg).\n\n"
        f"Conversation History:\n{conversation_context}\n\n"
        f"Document Context:\n{document_context}\n\n"
        f"Current Question: {query}\n\n"
        "Answer:"
    )
    
    # Print or log prompt for debugging if desired
    print("Generated prompt:\n", prompt)
    


    # Generate the response using the LLM.
    response = llm(
        prompt,
        max_tokens=1000,
        temperature=0.5
    )
    return response
    





