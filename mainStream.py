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
from fastapi.middleware.cors import CORSMiddleware


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

@app.get("/query")
async def query_index(query: str ):
    try:
        response_stream = query_engine.query(query)

        async def event_stream():
            for text in response_stream.response_gen:
                yield f"data: {text}\n\n"
            # Send a special message or marker to indicate the end of the stream
            yield "data: END_OF_STREAM\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")
        
    except Exception as e:
        logging.error(f"Error during query processing: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={"message": "LLM API is currently unavailable.", "error": str(e)}
        )

# Add CORS middleware to allow specific origins (or use '*' for all origins)
origins = [
   
     "*", # Allow all origins
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ... [rest of your code]

# The main function remains unchanged
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
