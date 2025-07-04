import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI(title="AI-Ayurveda Backend")
@app.get("/")

def read_root():

    return {"message": "Welcome to AI Ayurveda backend!"}
# Enable CORS to allow frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request model
class PromptRequest(BaseModel):
    prompt: str

# Initialize the transformer model and tokenizer
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Create a HuggingFace pipeline for text generation
hf_pipeline = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=200,
    device=-1  # CPU; use 0 for GPU if available
)

# Create a LangChain LLM using the HuggingFace pipeline
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# Define a prompt template for Ayurvedic context
# Updated Prompt Template
from langchain_core.prompts import PromptTemplate  # Updated import if you refactor

template = """
You are an expert Ayurvedic doctor. Based on ancient Ayurvedic texts, provide **precise and effective home remedies** using herbs, diet, yoga, and dosha balance for the given condition.

Only reply with Ayurvedic advice. If the query is not related to Ayurveda, say: "Please ask a question related to Ayurvedic remedies."

Question: {question}

Answer:
"""
prompt = PromptTemplate(template=template, input_variables=["question"])


# Create the LangChain
llm_chain = LLMChain(prompt=prompt, llm=llm)

# API endpoint to handle prompts
@app.post("/api/ayurveda")
async def process_prompt(request: PromptRequest):
    try:
        # Run the LangChain with the user's prompt
        response = llm_chain.run(question=request.prompt)
        return {"response": response.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing prompt: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}
