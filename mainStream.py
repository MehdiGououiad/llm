from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse

from pydantic import BaseModel
import os.path
import llama_index

from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    ServiceContext,
    load_index_from_storage,
    set_global_service_context,
    PromptTemplate
)
from llama_index.embeddings import HuggingFaceEmbedding


import logging
import sys


# Set global handler for LLaMA index
llama_index.set_global_handler("simple")

# Initialize FastAPI app
app = FastAPI()

# Define directory for persisting index
PERSIST_DIR = "./storage"

# Initialize embedding model
embed_model = HuggingFaceEmbedding(model_name="OrdalieTech/Solon-embeddings-large-0.1")

# Create service context with embedding model
service_context = ServiceContext.from_defaults(embed_model=embed_model)
set_global_service_context(service_context)

# Load or create the index
if not os.path.exists(PERSIST_DIR):
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

# Initialize query engine from index
query_engine = index.as_query_engine(streaming=True, similarity_top_k=2)

# Define custom prompt template
qa_prompt_tmpl_str = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge,Some rules to follow: 1. Avoid statements like 'Based on the context, ...' or 'The context information ...' or anything along those lines. "
    "answer the query in french and but remember you are chatbot trained on rh questions so always put that in perspective . you are named Rhym a chatbot created by the innovation team at BMCI  \n"
    "Query: {query_str}\n"
    "Answer: "
)
qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

# Update query engine with custom prompt template
query_engine.update_prompts(
    {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
)
# Define Pydantic model for query requests
class Query(BaseModel):
    text: str

# Define FastAPI endpoint for querying
@app.post("/query")
async def query_index(query: Query):
    try:
        response_stream = query_engine.query(query.text)

        # Define an asynchronous generator function
        async def generate():
           for text in response_stream.response_gen:
                # do something with text as they arrive.
                print(f"Streaming chunk: {text}")  # Simple print statement for demonstration

                yield text
                pass

        return StreamingResponse(generate(), media_type="text/plain")
        
    except Exception as e:
        # Log the exception for debugging purposes
        logging.error(f"Error during query processing: {str(e)}")
        # Return a JSON response with status code 503
        return JSONResponse(
            status_code=503,
            content={"message": "LLM API is currently unavailable.", "error": str(e)}
        )

# Main function to run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
