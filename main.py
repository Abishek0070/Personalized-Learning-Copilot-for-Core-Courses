from fastapi import (
    FastAPI, UploadFile, File, Body, Depends,
    HTTPException, status
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
import os, base64, tempfile, fitz
from datetime import datetime, timedelta, date
from typing import List, Dict, Optional
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlmodel import Session, select, create_engine, SQLModel
from jose import JWTError, jwt
from passlib.context import CryptContext
from dotenv import load_dotenv
from models import Student, MasteryRecord, StudyPlan, ChatHistory
import redis
import json

# Local Imports
from models import Student, MasteryRecord
from security import hash_password, verify_password

# LangChain Imports
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SECRET_KEY = "CHANGE_THIS_SECRET"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# --- Database & Auth Setup ---
sqlite_url = "sqlite:///database.db"
engine = create_engine(sqlite_url, connect_args={"check_same_thread": False})
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

app = FastAPI(title="Agentic Learning Copilot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session

# --- Auth Logic ---
def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme), session: Session = Depends(get_session)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        user = session.exec(select(Student).where(Student.username == username)).first()
        if user is None:
            raise HTTPException(status_code=401, detail="User not found")
        return user
    except JWTError:
        raise HTTPException(status_code=401, detail="Could not validate credentials")

# --- Schemas ---
class SignupRequest(BaseModel):
    username: str
    password: str
    full_name: str | None = None

class ChatRequest(BaseModel):
    query: str
    vector_id: str

class PlanRequest(BaseModel):
    query: str
    vector_id: str
    exam_date: date
    syllabus_text: str

class DailyTask(BaseModel):
    day: int
    topic: str
    description: str

class StudyPlanAI(BaseModel):
    plan: List[DailyTask]
    total_days: int
    motivation_quote: str

class StudyDay(BaseModel):
    day: int
    topic: str
    description: str

class StudyPlanModel(BaseModel):
    plan: List[StudyDay]


class Question(BaseModel):
    question: str
    options: List[str]
    answer: str
    explanation: Optional[str] = "Refer to the textbook for more details."

class Quiz(BaseModel):
    title: str
    questions: List[Question]

# --- Endpoints ---
@app.post("/signup")
def signup(req: SignupRequest, session: Session = Depends(get_session)):
    statement = select(Student).where(Student.username == req.username)
    if session.exec(statement).first():
        raise HTTPException(status_code=400, detail="Username already taken")
    
    new_user = Student(
        username=req.username,
        full_name=req.full_name,
        hashed_password=hash_password(req.password)
    )
    session.add(new_user)
    session.commit()
    return {"message": "User created successfully"}

@app.post("/login")
def login(form: OAuth2PasswordRequestForm = Depends(), session: Session = Depends(get_session)):
    user = session.exec(select(Student).where(Student.username == form.username)).first()
    if not user or not verify_password(form.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = create_access_token({"sub": user.username})
    return {"access_token": token, "token_type": "bearer"}

# ... Keep your process_multimodal_pdf, upload, and chat logic here ...

# =========================
# CORE API
# =========================
@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    user=Depends(get_current_user)
):
    # 1. Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        path = tmp.name

    # 2. Extract text
    raw_text = process_multimodal_pdf(path)
    if not raw_text:
        return {"error": "Failed to extract text from PDF"}

    # 3. BETTER SPLITTING: 
    # Increased chunk_size to 1500 and overlap to 200.
    # This keeps paragraphs together so the AI doesn't lose the "point" of a sentence.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, 
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    
    # Adding metadata (filename) helps the vector store stay organized
    docs = text_splitter.split_documents([
        Document(page_content=raw_text, metadata={"source": file.filename})
    ])

    # 4. Create Vector Store
    # We use CPU-friendly embeddings (all-MiniLM-L6-v2)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vs = FAISS.from_documents(docs, embeddings)

    # 5. Save to Disk
    user_index_path = f"index_{user.username}"
    vs.save_local(user_index_path)

    # Clean up
    os.remove(path)

    return {
        "message": "Success", 
        "index_name": user_index_path,
        "chunks_created": len(docs) # Useful for debugging in Streamlit
    }

import fitz  # This is PyMuPDF
import os

def process_multimodal_pdf(file_path: str):
    """
    Extracts text from a PDF. 
    (You can extend this to handle images/tables later)
    """
    text_content = ""
    try:
        # Open the PDF file
        doc = fitz.open(file_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text_content += page.get_text()
        doc.close()
        return text_content
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return None

def get_vector_store(username, embeddings):
    # The folder name we created in /upload
    index_path = f"index_{username}" 
    if os.path.exists(index_path):
        return FAISS.load_local(
            index_path, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
    return None

from sqlmodel import Session, select
from models import ChatHistory
r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)  # Ensure this is imported from your models.py

@app.post("/chat")
def chat(
    req: ChatRequest, 
    user=Depends(get_current_user), 
    session: Session = Depends(get_session)
):
    # 1. LOAD SHORT-TERM MEMORY FROM REDIS (Sub-millisecond)
    redis_key = f"chat_history:{user.username}"
    # Retrieve last 5 messages
    raw_history = r.lrange(redis_key, 0, 4) 
    
    chat_context = ""
    for entry in reversed(raw_history):
        data = json.loads(entry)
        chat_context += f"User: {data['q']}\nAI: {data['a']}\n"

    # 2. RAG RETRIEVAL (Document Context)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vs = get_vector_store(user.username, embeddings)
    
    if not vs:
        raise HTTPException(status_code=404, detail="PDF not found.")

    docs = vs.max_marginal_relevance_search(req.query, k=5, fetch_k=20)
    pdf_context = "\n---\n".join(d.page_content for d in docs)

    # 3. CONTEXT-AWARE PROMPT
    prompt = f"""
    You are a helpful AI Study Assistant.
    
    PREVIOUS CONVERSATION:
    {chat_context}

    PDF DOCUMENT CONTEXT:
    {pdf_context}

    NEW QUESTION: 
    {req.query}
    """
    
    # 4. GET RESPONSE
    llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant", temperature=0.1)
    response = llm.invoke(prompt)
    ai_answer = response.content

    # 5. UPDATE REDIS (Short-Term Memory)
    new_memory = json.dumps({"q": req.query, "a": ai_answer})
    r.lpush(redis_key, new_memory)
    r.ltrim(redis_key, 0, 9)  # Keep only latest 10 messages to save RAM
    r.expire(redis_key, 3600) # Auto-delete memory after 1 hour of inactivity

    # 6. SAVE TO SQLITE (Long-Term History)
    new_chat = ChatHistory(student_id=user.id, question=req.query, answer=ai_answer)
    session.add(new_chat)
    session.commit()

    return {"answer": ai_answer}

def run_structured_agent(prompt: str, response_model):
    """
    Uses Groq to generate a structured response based on a Pydantic model.
    """
    # Initialize the LLM
    llm = ChatGroq(
        model="llama-3.1-8b-instant", # Or your preferred Groq model
        temperature=0.3
    )
    
    # Bind the LLM to your Pydantic model (StudyPlan)
    structured_llm = llm.with_structured_output(response_model)
    
    # Invoke and return the typed object
    return structured_llm.invoke(prompt)

@app.post("/plan")
def generate_plan(
    req: PlanRequest, 
    user=Depends(get_current_user),
    session: Session = Depends(get_session)  # Added this missing dependency
):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vs = get_vector_store(user.username, embeddings)
    
    if not vs:
        raise HTTPException(status_code=404, detail="No PDF index found.")

    # 1. Retrieval
    search_query = "chapters, sections, and topics"
    docs = vs.similarity_search(search_query, k=5)
    real_context = "\n".join([d.page_content for d in docs])

    # 2. Timeline
    days_remaining = max(1, (req.exam_date - date.today()).days)

    # 3. Prompt (Simplified to avoid 400 errors)
    # We explicitly ask for the 'plan' key to match your StudyPlanModel
    prompt = f"""
    Create a {days_remaining}-day study schedule based ONLY on the text below.
    
    TEXT:
    {real_context}

    Output a JSON object with a 'plan' list. Each item in 'plan' must have:
    'day' (int), 'topic' (str), and 'description' (str).
    """

 # 4. Generate AI Result
    plan_result = run_structured_agent(prompt, StudyPlanAI) 

    try:
        # This StudyPlan now correctly refers to your SQLModel from models.py
        new_db_plan = StudyPlan(
            student_id=user.id,
            plan=[task.dict() for task in plan_result.plan],
            total_days=plan_result.total_days,
            motivation_quote=plan_result.motivation_quote
        )
        session.add(new_db_plan)
        session.commit()
    except Exception as e:
        print(f"Database error: {e}")

    return plan_result

@app.post("/quiz")
def generate_quiz(req: ChatRequest, user=Depends(get_current_user)):
    # 1. Initialize embeddings and load the store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vs = get_vector_store(user.username, embeddings)
    
    if not vs:
        raise HTTPException(status_code=404, detail="No PDF found.")

    # 2. RETRIEVE relevant chunks (Crucial step!)
    search_term = req.query if req.query else "key concepts"
    docs = vs.similarity_search(search_term, k=4)
    
    # 3. DEFINE 'context' before using it in the prompt
    context = "\n---\n".join([d.page_content for d in docs])

    # 4. Now use it in the prompt
    prompt = f"""
    Create a 3-question quiz based ONLY on this text:
    {context}
    """

    return run_structured_agent(prompt, Quiz)

@app.get("/my-history")
def get_history(user=Depends(get_current_user), session: Session = Depends(get_session)):
    # Fetch all chats for this user
    chats = session.exec(select(ChatHistory).where(ChatHistory.student_id == user.id)).all()
    
    # Fetch the most recent study plan
    plan = session.exec(
        select(StudyPlan)
        .where(StudyPlan.student_id == user.id)
        .order_by(StudyPlan.id.desc())
    ).first()
    
    return {
        "chats": chats,
        "latest_plan": plan
    }


@app.on_event("startup")
def on_startup():
    from models import SQLModel
    # This recreates the .db file with the 'hashed_password' column
    SQLModel.metadata.create_all(engine)
