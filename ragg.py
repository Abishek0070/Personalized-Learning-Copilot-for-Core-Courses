import streamlit as st
from datetime import datetime
import requests

API_URL = "http://127.0.0.1:8000"

# ---------------- SESSION STATE ----------------
# Keep your existing initialization
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "token" not in st.session_state:
    st.session_state.token = None
if "username" not in st.session_state:
    st.session_state.username = None
if "vector_id" not in st.session_state:
    st.session_state.vector_id = None
if "extracted_text" not in st.session_state:
    st.session_state.extracted_text = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- NEW: SYNC FUNCTION ---
def sync_user_data():
    """Fetches saved history and plans from the database on login."""
    headers = {"Authorization": f"Bearer {st.session_state.token}"}
    try:
        # 1. Fetch Chat History and Latest Plan
        # Note: Ensure you have an endpoint that returns both or call them separately
        res = requests.get(f"{API_URL}/my-history", headers=headers)
        if res.status_code == 200:
            data = res.json()
            st.session_state.chat_history = data.get("chats", [])
            
            # Restore study plan if it exists
            latest_plan = data.get("latest_plan")
            if latest_plan:
                # If your backend stores it as {'plan': [...]}, extract the list
                st.session_state.study_plan = latest_plan.get("plan", [])
    except Exception as e:
        st.error(f"Sync failed: {e}")

def load_saved_data():
    headers = {"Authorization": f"Bearer {st.session_state.token}"}
    res = requests.get(f"{API_URL}/my-history", headers=headers)
    
    if res.status_code == 200:
        data = res.json()
        # Restore chat history
        st.session_state.chat_history = data.get("chats", [])
        # Restore the study plan
        latest_plan = data.get("latest_plan")
        if latest_plan:
            st.session_state.study_plan = latest_plan.get("plan", [])

# ---------------- AUTH UI ----------------
if not st.session_state.logged_in:
    st.title("🚀 Personalized Learning Copilot")
    auth_mode = st.radio("Mode", ["Login", "Sign Up"])
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if auth_mode == "Sign Up":
        full_name = st.text_input("Full Name")
        if st.button("Register"):
            res = requests.post(f"{API_URL}/signup", json={"username": username, "password": password, "full_name": full_name})
            if res.status_code == 200:
                st.success("Signup successful! Switch to Login.")
            else:
                st.error(res.text)
    else:
        if st.button("Login"):
            res = requests.post(f"{API_URL}/login", data={"username": username, "password": password}, headers={"Content-Type": "application/x-www-form-urlencoded"})
            if res.status_code == 200:
                data = res.json()
                st.session_state.logged_in = True
                st.session_state.token = data["access_token"]
                st.session_state.username = username
                
                # TRIGGER SYNC IMMEDIATELY
                sync_user_data()
                load_saved_data()
                st.rerun()
            else:
                st.error("Invalid credentials")

# ---------------- DASHBOARD ----------------
else:
    headers = {"Authorization": f"Bearer {st.session_state.token}"}
    
    # Sidebar with History
    with st.sidebar:
        st.title(f"👤 {st.session_state.username}")
        if st.button("Logout"):
            st.session_state.clear()
            st.rerun()
        
        st.divider()
        st.subheader("📜 Recent Chats")
        # Display past questions as a simple list
        for chat in st.session_state.chat_history[-5:]:
            st.caption(f"Q: {chat['question'][:40]}...")

    st.title("📚 Student Dashboard")

    # --- 1. UPLOAD PDF ---
    with st.expander("📄 Step 1: Upload Study Material", expanded=not st.session_state.vector_id):
        pdf = st.file_uploader("Upload PDF", type=["pdf"])
        if pdf and st.button("Process PDF"):
            with st.spinner("Analyzing PDF..."):
                res = requests.post(f"{API_URL}/upload", files={"file": pdf}, headers=headers)
                if res.status_code == 200:
                    data = res.json()
                    st.session_state.vector_id = data.get("index_name")
                    st.session_state.extracted_text = data.get("raw_text", "")
                    st.success("PDF Ready!")
                else:
                    st.error("Upload failed.")

    # --- 2. CHAT ---
    st.subheader("💬 Chat with AI Tutor")
    
    # Display current session messages
    for chat in st.session_state.chat_history[-3:]: # Show last 3 messages
        with st.chat_message("user"): st.write(chat["question"])
        with st.chat_message("assistant"): st.write(chat["answer"])

    query = st.text_input("Ask a question...")
    if st.button("Ask") and query:
        res = requests.post(f"{API_URL}/chat", json={"query": query, "vector_id": st.session_state.vector_id}, headers=headers)
        if res.status_code == 200:
            ans = res.json()["answer"]
            # Append to session state so it shows up immediately
            st.session_state.chat_history.append({"question": query, "answer": ans})
            st.rerun()

    st.divider()

    # --- 3. PLANNER ---
    st.subheader("📅 Smart Study Planner")
    # If a plan was synced from DB, it will show up here automatically
    if "study_plan" in st.session_state and st.session_state.study_plan:
        with st.expander("View Your Saved Plan", expanded=False):
            for entry in st.session_state.study_plan:
                with st.container(border=True):
                    st.markdown(f"**Day {entry['day']}: {entry['topic']}**")
                    st.write(entry['description'])

    target_date = st.date_input("Exam Date", min_value=datetime.today())
    if st.button("Generate New Plan"):
        payload = {
            "query": "Generate plan",
            "vector_id": st.session_state.vector_id,
            "exam_date": str(target_date),
            "syllabus_text": st.session_state.extracted_text[:5000]
        }
        res = requests.post(f"{API_URL}/plan", json=payload, headers=headers)
        if res.status_code == 200:
            data = res.json()
            # Handle both list and dict responses
            st.session_state.study_plan = data["plan"] if isinstance(data, dict) and "plan" in data else data
            st.success("New Plan Created and Saved!")
            st.rerun()

    st.divider()

    # --- 4. QUIZ ---
    
    st.subheader("📝 Practice Quiz")
    if st.button("Generate Quiz"):
        res = requests.post(f"{API_URL}/quiz", json={"query": "Core concepts", "vector_id": st.session_state.vector_id}, headers=headers)
        if res.status_code == 200:
            st.session_state.current_quiz = res.json()["questions"]
    
    if "current_quiz" in st.session_state:
        for i, q in enumerate(st.session_state.current_quiz):
            st.markdown(f"**Q{i+1}: {q['question']}**")
            choice = st.radio("Select answer:", q['options'], key=f"q_{i}")
            if st.button(f"Submit Q{i+1}", key=f"btn_{i}"):
                if choice == q['answer']: st.success("Correct!")
                else: st.error(f"Wrong! Answer: {q['answer']}")
