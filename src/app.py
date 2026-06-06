import os
import sys

# Define base directory as the directory containing this file (src/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Add parent directory to python path for cloud deployment imports
sys.path.append(os.path.dirname(BASE_DIR))

import re
import ast
import pickle
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from src.parser import parse_resume, extract_skills_from_text, parse_education_level, parse_required_experience

# Set Streamlit Page Config
st.set_page_config(
    page_title="TalentVantage AI | Intelligent Talent Acquisition",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ----------------- Theme CSS Generator (Strictly Dark-Luxury) -----------------
def get_theme_css():
    return """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Outfit:wght@500;600;700;800&display=swap');
        
        :root {
            --bg-main: #080c14;
            --bg-card: #0d1527;
            --border: rgba(129, 140, 248, 0.15);
            --text-high: #f3f4f6;
            --text-mid: #9ca3af;
            --accent: #818cf8;
            --success: #10b981;
            --error: #ef4444;
            --header-bg: rgba(13, 21, 39, 0.7);
            --shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
        }
        
        /* Diffused Background Glow Orb */
        body::before {
            content: "";
            position: absolute;
            top: 0;
            right: 0;
            width: 600px;
            height: 600px;
            background: radial-gradient(circle, rgba(129, 140, 248, 0.15) 0%, rgba(129, 140, 248, 0) 70%);
            pointer-events: none;
            z-index: 0;
        }
        
        .stApp {
            background-color: var(--bg-main) !important;
            color: var(--text-high) !important;
            font-family: 'Inter', -apple-system, sans-serif !important;
        }
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Outfit', sans-serif !important;
            color: var(--text-high) !important;
            font-weight: 700 !important;
        }
        
        /* Main header logo and title */
        .main-title {
            background: linear-gradient(90deg, #c084fc, #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3rem;
            font-weight: 800;
            margin-bottom: 5px;
            letter-spacing: -0.03em;
        }
        .subtitle {
            color: var(--text-mid);
            font-size: 1.1rem;
            margin-bottom: 30px;
        }
        .glass-header-panel {
            text-align: center;
            background: rgba(13, 21, 39, 0.45) !important;
            border: 1px solid rgba(129, 140, 248, 0.2) !important;
            backdrop-filter: blur(12px) saturate(180%) !important;
            -webkit-backdrop-filter: blur(12px) saturate(180%) !important;
            border-radius: 16px !important;
            padding: 25px 40px !important;
            margin: 20px auto 35px auto !important;
            max-width: 800px !important;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important;
        }
        .glass-header-panel .subtitle {
            margin-bottom: 0 !important;
        }
        
        /* Card Containers targeted via markers (resolves container fallout issues) */
        div[data-testid="stVerticalBlock"]:has(div.step1-card-marker):not(:has(div[data-testid="stVerticalBlock"] div.step1-card-marker)),
        div[data-testid="stVerticalBlock"]:has(div.step2-card-marker):not(:has(div[data-testid="stVerticalBlock"] div.step2-card-marker)) {
            background-color: var(--bg-card) !important;
            border: 1px solid var(--border) !important;
            border-radius: 16px !important;
            padding: 24px !important;
            margin-bottom: 24px !important;
            box-shadow: var(--shadow) !important;
        }
        
        .metric-card {
            background-color: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
            box-shadow: var(--shadow);
            text-align: center;
            transition: transform 0.2s ease, border-color 0.2s ease;
        }
        .metric-card:hover {
            transform: translateY(-2px);
            border-color: var(--accent);
        }
        .metric-val {
            font-size: 2.2rem;
            font-weight: 700;
            color: var(--accent);
            margin-bottom: 5px;
        }
        .metric-lbl {
            font-size: 0.85rem;
            color: var(--text-mid);
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        /* Badge tags */
        .tag-match {
            background-color: rgba(16, 185, 129, 0.15);
            color: var(--success);
            border: 1px solid rgba(16, 185, 129, 0.3);
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
            display: inline-block;
            margin: 2px;
        }
        .tag-gap {
            background-color: rgba(239, 68, 68, 0.15);
            color: var(--error);
            border: 1px solid rgba(239, 68, 68, 0.3);
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
            display: inline-block;
            margin: 2px;
        }
        
        /* Kanban Board */
        .kanban-col {
            background-color: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 15px;
            min-height: 480px;
        }
        .kanban-header {
            text-align: center;
            background-color: var(--bg-main);
            border-bottom: 2px solid var(--border);
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 15px;
            font-weight: 600;
            font-size: 1.05rem;
            color: var(--text-high);
        }
        .kanban-card {
            background-color: var(--bg-main);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 12px;
            transition: border-color 0.2s ease, transform 0.2s ease;
        }
        .kanban-card:hover {
            border-color: var(--accent);
            transform: translateY(-1px);
        }
        
        /* Resume viewer */
        .resume-viewer {
            background-color: var(--bg-main);
            border: 1px solid var(--border);
            padding: 20px;
            border-radius: 8px;
            max-height: 500px;
            overflow-y: scroll;
            white-space: pre-wrap;
            font-family: 'Inter', sans-serif;
            color: var(--text-high);
            line-height: 1.6;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
            background: transparent;
        }
        .stTabs [data-baseweb="tab"] {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            color: var(--text-mid);
            padding: 10px 20px;
            font-weight: 500;
        }
        .stTabs [aria-selected="true"] {
            background: var(--accent) !important;
            color: white !important;
            border: none !important;
        }
        
        /* Outlined Action Buttons inside columns */
        div[data-testid="stHorizontalBlock"] button {
            background: transparent !important;
            border: 1px solid var(--border) !important;
            color: var(--text-high) !important;
            box-shadow: none !important;
            font-weight: 500 !important;
        }
        
        /* Primary Button Styling */
        div[data-testid="stButton"] button {
            background-color: var(--accent) !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 8px 16px !important;
            font-weight: 600 !important;
            transition: background-color 0.2s ease !important;
        }
        div[data-testid="stButton"] button:hover {
            background-color: var(--text-high) !important;
            color: var(--bg-main) !important;
        }
        
        /* Fixed position layout floating chatbot button in the right corner from start */
        div[data-testid="stVerticalBlock"]:has(div.chat-btn-marker):not(:has(div[data-testid="stVerticalBlock"] div.chat-btn-marker)) {
            position: fixed !important;
            bottom: 30px !important;
            right: 30px !important;
            z-index: 99999 !important;
            width: auto !important;
        }
        div.chat-btn-marker + div[data-testid="stButton"] button {
            background: linear-gradient(135deg, #818cf8, #6366f1) !important;
            color: white !important;
            border-radius: 50px !important;
            padding: 12px 24px !important;
            font-size: 1rem !important;
            font-weight: 600 !important;
            border: none !important;
            box-shadow: 0 4px 15px rgba(129, 140, 248, 0.4) !important;
            cursor: pointer;
        }
        
        /* Floating Chatbot Dialogue Window (Zomato/Swiggy style) */
        div[data-testid="stVerticalBlock"]:has(div.chat-window-marker):not(:has(div[data-testid="stVerticalBlock"] div.chat-window-marker)) {
            position: fixed !important;
            bottom: 95px !important;
            right: 30px !important;
            width: 380px !important;
            height: 560px !important;
            background-color: var(--bg-card) !important;
            border: 1px solid var(--border) !important;
            border-radius: 16px !important;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5) !important;
            z-index: 99998 !important;
            padding: 20px !important;
            display: block !important;
            overflow-y: auto !important;
        }
        
        /* Fixed internal scroll container for Chat bubble list */
        div[data-testid="stVerticalBlock"]:has(div.chat-messages-scroll):not(:has(div[data-testid="stVerticalBlock"] div.chat-messages-scroll)) {
            height: auto !important;
            max-height: none !important;
            overflow: visible !important;
            border: 1px solid var(--border) !important;
            border-radius: 8px !important;
            padding: 10px !important;
            background-color: var(--bg-main) !important;
            margin-bottom: 10px !important;
        }
        
        /* Custom glass header layout */
        div[data-testid="stVerticalBlock"]:has(div.glass-header-marker):not(:has(div[data-testid="stVerticalBlock"] div.glass-header-marker)) {
            background: var(--header-bg) !important;
            border: 1px solid var(--border) !important;
            backdrop-filter: blur(12px) !important;
            border-radius: 50px !important;
            padding: 10px 30px !important;
            margin: 15px auto 40px auto !important;
            max-width: 1000px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: space-between !important;
            box-shadow: var(--shadow) !important;
        }
        .header-logo {
            font-family: 'Outfit', sans-serif;
            font-weight: 700;
            font-size: 1.25rem;
            background: linear-gradient(90deg, #c084fc, #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .header-nav {
            font-family: 'Outfit', sans-serif;
            font-weight: 500;
            font-size: 0.95rem;
            letter-spacing: 1px;
        }
        .header-nav a {
            color: var(--text-mid);
            text-decoration: none;
            transition: color 0.2s ease;
        }
        .header-nav a:hover {
            color: var(--accent);
        }
        
        /* Close chat button marker target */
        div.close-chat-marker + div[data-testid="stButton"] button {
            background: transparent !important;
            border: none !important;
            color: var(--text-mid) !important;
            font-size: 1.2rem !important;
            padding: 0 !important;
            box-shadow: none !important;
            width: auto !important;
        }
        div.close-chat-marker + div[data-testid="stButton"] button:hover {
            color: var(--error) !important;
        }
        
        [data-testid="collapsedControl"] {
            display: none;
        }
        header {
            visibility: hidden;
        }
        
        /* Chat bubble styling components */
        .chat-bubbles-container {
            display: flex;
            flex-direction: column;
            gap: 10px;
            padding: 5px;
        }
        .chat-row {
            display: flex;
            flex-direction: column;
            max-width: 85%;
            margin-bottom: 8px;
        }
        .user-row {
            align-self: flex-end;
            align-items: flex-end;
            margin-left: auto;
        }
        .assistant-row {
            align-self: flex-start;
            align-items: flex-start;
            margin-right: auto;
        }
        .chat-sender {
            font-size: 0.72rem;
            color: var(--text-mid);
            margin-bottom: 2px;
            margin-left: 4px;
            margin-right: 4px;
            font-family: 'Inter', sans-serif;
        }
        .chat-bubble {
            padding: 10px 14px;
            border-radius: 12px;
            font-size: 0.85rem;
            line-height: 1.4;
            font-family: 'Inter', sans-serif;
            word-break: break-word;
        }
        .user-bubble {
            background-color: var(--accent) !important;
            color: white !important;
            border-top-right-radius: 2px;
        }
        .assistant-bubble {
            background-color: #1e293b !important;
            color: #f3f4f6 !important;
            border: 1px solid rgba(129, 140, 248, 0.15);
            border-top-left-radius: 2px;
        }
        .chat-status-dot {
            display: inline-block;
            width: 8px;
            height: 8px;
            background-color: #10b981;
            border-radius: 50%;
            box-shadow: 0 0 8px #10b981;
        }
        
        /* Streamlit form container override */
        div[data-testid="stForm"] {
            border: none !important;
            background: transparent !important;
            padding: 0 !important;
            margin-top: 18px !important;
        }
        
        /* Form submit button styled as a round/square send icon */
        div[data-testid="stForm"] div[data-testid="stFormSubmitButton"] button {
            background-color: var(--accent) !important;
            color: white !important;
            border-radius: 8px !important;
            padding: 8px !important;
            font-size: 1.15rem !important;
            font-weight: 600 !important;
            width: 100% !important;
            height: 42px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            box-shadow: none !important;
            border: none !important;
        }
        
        /* Preset buttons styling */
        div.preset-btn-marker + div[data-testid="stButton"] button {
            background-color: #1e293b !important;
            border: 1px solid rgba(129, 140, 248, 0.2) !important;
            color: #cbd5e1 !important;
            border-radius: 20px !important;
            font-size: 0.8rem !important;
            padding: 6px 12px !important;
            transition: all 0.2s ease !important;
            width: 100% !important;
            box-shadow: none !important;
        }
        div.preset-btn-marker + div[data-testid="stButton"] button:hover {
            background-color: var(--accent) !important;
            color: white !important;
            border-color: var(--accent) !important;
        }
    </style>
    """

# ----------------- Cache Assets Loader -----------------
@st.cache_resource
def load_assets():
    try:
        with open(os.path.join(BASE_DIR, '../models/skills_vocab.pkl'), 'rb') as f:
            skills_vocab = pickle.load(f)
        with open(os.path.join(BASE_DIR, '../models/tfidf_vectorizer.pkl'), 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
        with open(os.path.join(BASE_DIR, '../models/match_model.pkl'), 'rb') as f:
            match_model = pickle.load(f)
        with open(os.path.join(BASE_DIR, '../models/metrics.pkl'), 'rb') as f:
            model_metrics = pickle.load(f)
        return skills_vocab, tfidf_vectorizer, match_model, model_metrics
    except Exception as e:
        st.error(f"Failed to load trained model assets. Ensure `model.py` was executed. Details: {e}")
        return None, None, None, None

@st.cache_data
def load_bert_embeddings():
    path = os.path.join(BASE_DIR, '../models/bert_embeddings.pkl')
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return {}

@st.cache_resource
def load_bert_model():
    import torch
    torch.set_num_threads(4)
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    model.eval()
    return tokenizer, model

skills_vocab, tfidf_vectorizer, match_model, model_metrics = load_assets()
bert_embeddings = load_bert_embeddings()

# Load Demo Job Descriptions
def load_demo_jobs():
    return {
        "Senior Software Engineer": {
            "title": "Senior Software Engineer",
            "text": "Senior Software Engineer\nWe are looking for a Senior Software Engineer with strong expertise in Java, Spring Boot, MySQL, and cloud services (AWS or Azure). The ideal candidate must have at least 5 years of software development experience, design robust microservices architecture, and participate in unit testing, debugging, and continuous integration. Required skills: Java, Spring Boot, SQL, AWS, Docker, Microservices, Git, JUnit.",
            "skills": "Java, Spring Boot, SQL, AWS, Docker, Microservices, Git, JUnit",
            "experience": 5,
            "education": "Bachelor's Degree in Computer Science"
        },
        "Machine Learning Engineer": {
            "title": "Machine Learning Engineer",
            "text": "Machine Learning Engineer / Data Scientist\nSeeking an ML Engineer with 3+ years of experience training predictive models. Candidate must be skilled in Python, Scikit-learn, TensorFlow, PyTorch, Pandas, and SQL. Responsibilities include building data pipelines, statistical data analysis, model deployment, and NLP keyword extraction. Required skills: Python, Machine Learning, TensorFlow, PyTorch, Scikit-learn, Pandas, NLP, SQL.",
            "skills": "Python, Machine Learning, TensorFlow, PyTorch, Scikit-learn, Pandas, NLP, SQL",
            "experience": 3,
            "education": "Master's Degree in Computer Science or Statistics"
        },
        "Senior Business Analyst": {
            "title": "Senior Business Analyst (BSA)",
            "text": "Senior Business Systems Analyst\nWe need a Senior Business Analyst with a minimum of 4 years of experience gathering requirements, writing detailed user stories, creating process flow charts in Visio, and coordinating between engineering teams and stakeholders. Must have strong understanding of Agile / Scrum methodologies. Required skills: Requirement Gathering, Agile, Scrum, Jira, MS Visio, Business Process Mapping, User Stories, UAT Testing.",
            "skills": "Requirement Gathering, Agile, Scrum, Jira, MS Visio, Business Process Mapping, User Stories, UAT Testing",
            "experience": 4,
            "education": "Bachelor's Degree"
        },
        "Technical Project Manager / Scrum Master": {
            "title": "Technical Project Manager (PM / Scrum Master)",
            "text": "Technical Program Manager / Scrum Master\nWe are hiring a Scrum Master and Project Manager to lead agile sprint teams. Requirements: 6+ years experience, PMP or Scrum Master (CSM) certification, expertise in Jira, team leadership, risk management, and software development lifecycles. Required skills: Project Management, Agile, Scrum, Scrum Master, Jira, PMP, Risk Management, Team Leadership, SDLC.",
            "skills": "Project Management, Agile, Scrum, Scrum Master, Jira, PMP, Risk Management, Team Leadership, SDLC",
            "experience": 6,
            "education": "Bachelor's or Master's Degree"
        }
    }

demo_jobs = load_demo_jobs()

# Custom Candidate Redaction for Blind Screening
def redact_profile(profile, candidate_idx):
    return {
        'filename': profile['filename'],
        'filepath': profile['filepath'],
        'name': f"Candidate #{candidate_idx}",
        'email': "[REDACTED]@blind-screen.com",
        'phone': "[REDACTED]",
        'location': "Hidden (Blind Screening)",
        'education_degree': profile['education_degree'],
        'education_level': profile['education_level'],
        'experience_years': profile['experience_years'],
        'skills': profile['skills'],
        'text': profile['text']
    }

# Highlight keywords in Resume HTML
def highlight_resume_text(text, common_skills):
    if not text:
        return ""
    
    html_text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    sorted_skills = sorted(common_skills, key=len, reverse=True)
    for skill in sorted_skills:
        if len(skill.strip()) < 2:
            continue
        from src.parser import get_skill_pattern
        pattern = get_skill_pattern(skill)
        html_text = re.sub(
            pattern,
            lambda m: f'<span style="background-color: rgba(16, 185, 129, 0.25); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.4); padding: 1px 4px; border-radius: 4px; font-weight: 600;">{m.group(0)}</span>',
            html_text,
            flags=re.IGNORECASE
        )
        
    warning_words = ['gap', 'gaps', 'career break', 'unemployed', 'terminated', 'resigned', 'fired', 'contract ended', 'break in service']
    for word in warning_words:
        pattern = rf'\b({re.escape(word)}s?)\b'
        html_text = re.sub(
            pattern,
            lambda m: f'<span style="background-color: rgba(239, 68, 68, 0.25); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.4); padding: 1px 4px; border-radius: 4px; font-weight: 600;">{m.group(0)}</span>',
            html_text,
            flags=re.IGNORECASE
        )
        
    return html_text

# Bias Detection Keywords
def analyze_bias(jd_text):
    masculine_words = ['ninja', 'rockstar', 'guru', 'dominant', 'competitive', 'leader', 'decisive', 'assertive', 'active', 'analyse', 'drive', 'ambitious', 'force']
    feminine_words = ['collaborative', 'supportive', 'cooperative', 'empathy', 'interpersonal', 'team player', 'responsible', 'care', 'support', 'connect', 'share', 'trust']
    age_words = ['young', 'energetic', 'recent graduate', 'digital native', 'fresh', 'highly active', 'mature', 'seasoned', 'veteran']
    
    def count_matches(txt, word_list):
        if not txt:
            return []
        txt_lower = txt.lower()
        found = []
        for w in word_list:
            if re.search(rf'\b{re.escape(w)}\b', txt_lower):
                found.append(w)
        return found
        
    jd_masc = count_matches(jd_text, masculine_words)
    jd_fem = count_matches(jd_text, feminine_words)
    jd_age = count_matches(jd_text, age_words)
    
    res = {
        'jd_masculine': jd_masc,
        'jd_feminine': jd_fem,
        'jd_age': jd_age,
        'gender_balance': 'Neutral'
    }
    
    m_count = len(jd_masc)
    f_count = len(jd_fem)
    if m_count > f_count + 1:
        res['gender_balance'] = 'Slightly Masculine-Skewed' if m_count - f_count <= 2 else 'Masculine-Skewed'
    elif f_count > m_count + 1:
        res['gender_balance'] = 'Slightly Feminine-Skewed' if f_count - m_count <= 2 else 'Feminine-Skewed'
        
    return res

# Ollama Connection Helpers
def check_ollama_status(url="http://localhost:11434/"):
    try:
        response = requests.get(url, timeout=1.5)
        return response.status_code == 200
    except:
        return False

def get_ollama_models(url="http://localhost:11434/"):
    try:
        response = requests.get(f"{url.rstrip('/')}/api/tags", timeout=1.5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            return [m['name'] for m in models]
    except:
        pass
    return []

# BERT Embedding Helper
def get_bert_embedding(text, tokenizer, model):
    import torch
    
    # Truncate to 1500 characters to minimize CPU tokenizer overhead
    # since MPNet has a standard hard limit of 512 tokens (~1500 chars) anyway.
    if text and len(text) > 1500:
        text = text[:1500]
    
    # Mean Pooling
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    encoded_input = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    return sentence_embeddings[0].numpy()

# ----------------- Helper functions for Chatbot -----------------
def md_to_html(text):
    if not text:
        return ""
    # Escape html characters
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # Replace bold
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    # Replace bullet lists
    text = re.sub(r'^\s*-\s+(.*?)$', r'&bull; \1', text, flags=re.MULTILINE)
    # Replace newlines with breaks
    text = text.replace('\n', '<br>')
    return text

def render_chatbot(scored_candidates=None, jd_title="", jd_text=""):
    # Initialize session state for chatbot
    if 'chat_open' not in st.session_state:
        st.session_state.chat_open = True
        
    # Draw floating chat trigger button at bottom right (Targeted via chat-btn-marker)
    chat_btn_container = st.container()
    with chat_btn_container:
        st.markdown('<div class="chat-btn-marker"></div>', unsafe_allow_html=True)
        if st.button("💬 Chat", key="chat_toggle_btn"):
            st.session_state.chat_open = not st.session_state.chat_open
            st.rerun()
            
    # Draw floating chat window if open
    if st.session_state.chat_open:
        chat_container = st.container()
        with chat_container:
            st.markdown('<div class="chat-window-marker"></div>', unsafe_allow_html=True)
            
            # Connection status and config
            # Check if OLLAMA_URL is configured in secrets (Production/Cloud), otherwise default to Localhost (Development)
            if "OLLAMA_URL" in st.secrets:
                ollama_url = st.secrets["OLLAMA_URL"]
            else:
                ollama_url = "http://localhost:11434"
            ollama_online = check_ollama_status(ollama_url)
            
            hf_token = ""
            if hasattr(st.secrets, "get"):
                hf_token = st.secrets.get("HF_TOKEN", "")
            elif "HF_TOKEN" in st.secrets:
                hf_token = st.secrets["HF_TOKEN"]
            if not hf_token:
                hf_token = os.environ.get("HF_TOKEN", "")
                
            if ollama_online:
                status_color = "#10b981" # Green
                status_text = "Co-Pilot Online"
            elif hf_token:
                status_color = "#818cf8" # Indigo/Purple
                status_text = "Cloud Backup Active"
            else:
                status_color = "#ef4444" # Red
                status_text = "Service Offline"
                
            # Header with close button
            col_header, col_close = st.columns([5, 1])
            with col_header:
                st.markdown(f"""
                <div style="display:flex; align-items:center; gap:8px;">
                    <span class="chat-status-dot" style="background-color:{status_color}; box-shadow:0 0 8px {status_color};"></span>
                    <div>
                        <div style="font-weight:700; font-size:0.95rem; color:var(--text-high); font-family:'Outfit';">Talent Co-Pilot</div>
                        <div style="font-size:0.75rem; color:{status_color}; font-family:'Inter';">{status_text}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with col_close:
                st.markdown('<div class="close-chat-marker"></div>', unsafe_allow_html=True)
                if st.button("✕", key="close_chat_btn", help="Close chat"):
                    st.session_state.chat_open = False
                    st.rerun()
            
            # Show a quiet alert if both local and cloud services are offline
            if not ollama_online and not hf_token:
                st.markdown(
                    "<div style='background-color:rgba(239, 68, 68, 0.1); border:1px solid rgba(239, 68, 68, 0.2); padding: 8px; border-radius: 6px; font-size: 0.75rem; color: #ef4444; margin-bottom: 8px; font-family: Inter;'>"
                    "⚠️ Talent Co-Pilot is offline. Launch local backend service or configure the Cloud AI Fallback token in secrets to resume."
                    "</div>",
                    unsafe_allow_html=True
                )
                
            # Resolve selected model
            selected_model = "llama3.2:latest"
            if ollama_online:
                models = get_ollama_models(ollama_url)
                # Prioritize faster text-only models (like llama3.2, llama3, phi3, mistral, gemma) and avoid large multimodal models like llava by default
                if models:
                    pref_keywords = ["llama3.2", "llama3", "phi3", "mistral", "gemma", "qwen"]
                    found_pref = False
                    for kw in pref_keywords:
                        for m in models:
                            if kw in m.lower():
                                selected_model = m
                                found_pref = True
                                break
                        if found_pref:
                            break
                    if not found_pref:
                        non_llava = [m for m in models if "llava" not in m.lower()]
                        if non_llava:
                            selected_model = non_llava[0]
                        else:
                            selected_model = models[0]
            else:
                selected_model = "Qwen/Qwen2.5-7B-Instruct" # Backup cloud model

            # Determine candidate context automatically from Deep Applicant Inspector or default
            p_chat = None
            if scored_candidates:
                if 'rank_select' in st.session_state and st.session_state.rank_select:
                    selected_str = st.session_state.rank_select
                    for c in scored_candidates:
                        if c['profile']['name'] in selected_str:
                            p_chat = c['profile']
                            break
                if not p_chat:
                    p_chat = scored_candidates[0]['profile']
            else:
                p_chat = None
            
            # History container
            chat_history_key = f"chat_hist_{p_chat['filename']}" if p_chat else "chat_hist_general"
            if chat_history_key not in st.session_state:
                st.session_state[chat_history_key] = []
            
            # Professional welcoming greeting message (like Swiggy / Zomato chatbot)
            if not st.session_state[chat_history_key]:
                if p_chat:
                    greeting = f"Hi! I am your AI Talent Co-Pilot. I can help you summarize {p_chat['name']}'s profile, generate behavioral interview questions, or analyze skill gaps. What would you like to do?"
                else:
                    greeting = "Hi! I am your AI Talent Co-Pilot. I can help you screen resumes and answer recruitment queries. Please upload resumes to start analyzing specific candidates!"
                st.session_state[chat_history_key] = [
                    {
                        "role": "assistant",
                        "content": greeting
                    }
                ]
                
            # Scrollable chat messages container
            chat_scroll_area = st.container()
            with chat_scroll_area:
                st.markdown('<div class="chat-messages-scroll"></div>', unsafe_allow_html=True)
                
                # Render messages as custom chat bubbles in HTML
                chat_html = "<div class='chat-bubbles-container'>"
                for msg in st.session_state[chat_history_key]:
                    role_class = "user-bubble" if msg['role'] == "user" else "assistant-bubble"
                    sender_name = "You" if msg['role'] == "user" else "AI Talent Co-Pilot"
                    # Convert markdown to html for display
                    formatted_content = md_to_html(msg['content'])
                    chat_html += f'<div class="chat-row {"user-row" if msg["role"] == "user" else "assistant-row"}">'
                    chat_html += f'<div class="chat-sender">{sender_name}</div>'
                    chat_html += f'<div class="chat-bubble {role_class}">{formatted_content}</div>'
                    chat_html += '</div>'
                chat_html += "</div>"
                st.markdown(chat_html, unsafe_allow_html=True)
            
            # Preset Prompts
            preset_prompt = ""
            if p_chat:
                st.markdown("<div style='font-size:0.75rem; color:var(--text-mid); margin-bottom:5px; font-weight:600; font-family:Inter;'>SUGGESTED ACTIONS:</div>", unsafe_allow_html=True)
                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    st.markdown('<div class="preset-btn-marker"></div>', unsafe_allow_html=True)
                    if st.button("📝 Summarize", key="sum_btn", help="Summarize Candidate"):
                        preset_prompt = "Provide a 3-bullet point summary of this candidate's core qualifications, years of experience, and highest degree."
                with col_p2:
                    st.markdown('<div class="preset-btn-marker"></div>', unsafe_allow_html=True)
                    if st.button("❓ Interview Prep", key="q_btn", help="Generate Interview Questions"):
                        preset_prompt = "Generate 5 customized technical and behavioral interview questions for this candidate based on their work history."
            
            # Chat Input Form (custom styled to stay neatly at the bottom)
            with st.form(key="chat_input_form", clear_on_submit=True):
                col_in, col_send = st.columns([4.2, 0.8])
                with col_in:
                    user_input = st.text_input("", placeholder="Ask a question...", key="chat_msg_input", label_visibility="collapsed")
                with col_send:
                    submit_chat = st.form_submit_button("➤")
            
            final_prompt = preset_prompt if preset_prompt else (user_input if submit_chat else "")
            
            if final_prompt:
                st.session_state[chat_history_key].append({"role": "user", "content": final_prompt})
                
                # Fetch LLM response
                if p_chat:
                    # Construct a compact candidates summary to minimize prompt context size and speed up ingestion
                    c_skills = ", ".join(p_chat.get('skills', []))
                    c_exp = p_chat.get('experience_years', 0)
                    c_degree = p_chat.get('education_degree', 'N/A')
                    c_text_snippet = p_chat.get('text', '')[:1200]
                    
                    system_instruction = f"""You are Talent Co-Pilot, a professional recruitment assistant.
Analyzing applicant '{p_chat['name']}' for the role '{jd_title}'.

Candidate Profile:
- Experience: {c_exp} years
- Education: {c_degree}
- Skills: {c_skills}
- Summary: {c_text_snippet}

JD Summary: {jd_text[:800]}

Instruction: Provide extremely concise, direct answers (max 2-3 sentences or 3 bullet points). Do not include conversational filler, introductory remarks, or repeat context details."""
                else:
                    system_instruction = """You are Talent Co-Pilot, a professional recruitment assistant. 
Help the user understand how to use the Resume Screening application (upload resumes, select job descriptions, and view rankings).
Instruction: Keep your response extremely brief and direct (max 2-3 sentences)."""
                
                ollama_success = False
                ai_text = ""
                error_msg = ""
                
                if ollama_online:
                    try:
                        with st.spinner("AI is thinking (Local)..."):
                            response = requests.post(
                                f"{ollama_url.rstrip('/')}/api/chat",
                                json={
                                    "model": selected_model,
                                    "messages": [
                                        {
                                            "role": "user",
                                            "content": f"{system_instruction}\n\nQuestion: {final_prompt}"
                                        }
                                    ],
                                    "stream": False,
                                    "options": {
                                        "temperature": 0.3,
                                        "num_predict": 120
                                    }
                                },
                                timeout=5 # Fail fast to check cloud fallback
                            )
                            
                            if response.status_code == 200:
                                ai_text = response.json().get('message', {}).get('content', '')
                                ollama_success = True
                            else:
                                error_msg = f"Local engine returned status {response.status_code}."
                    except Exception as ex:
                        error_msg = str(ex)
                else:
                    error_msg = "Local connection offline."
                    
                # Fallback to Cloud Backup
                if not ollama_success:
                    if hf_token:
                        try:
                            with st.spinner("Local offline. Querying Cloud Backup..."):
                                hf_url = "https://api-inference.huggingface.co/v1/chat/completions"
                                hf_headers = {
                                    "Content-Type": "application/json",
                                    "Authorization": f"Bearer {hf_token}"
                                }
                                hf_payload = {
                                    "model": "Qwen/Qwen2.5-7B-Instruct",
                                    "messages": [
                                        {
                                            "role": "system",
                                            "content": system_instruction
                                        },
                                        {
                                            "role": "user",
                                            "content": final_prompt
                                        }
                                    ],
                                    "max_tokens": 120,
                                    "temperature": 0.3
                                }
                                
                                hf_res = requests.post(hf_url, json=hf_payload, headers=hf_headers, timeout=20)
                                if hf_res.status_code == 200:
                                    ai_text = hf_res.json()["choices"][0]["message"]["content"]
                                    ai_text += "\n\n*(⚡ Cloud Backup)*"
                                    ollama_success = True
                                elif hf_res.status_code == 401 or hf_res.status_code == 403:
                                    error_msg = "Cloud Backup unauthorized. Please check that a valid token is configured in application secrets."
                                else:
                                    error_msg = f"Cloud Backup status code: {hf_res.status_code}"
                        except Exception as hf_ex:
                            error_msg = f"Local offline ({error_msg}). Cloud Backup failed: {hf_ex}"
                    else:
                        error_msg = f"Local offline ({error_msg}). Cloud Backup is not configured (missing fallback token)."
                        
                if ollama_success:
                    st.session_state[chat_history_key].append({"role": "assistant", "content": ai_text})
                else:
                    st.session_state[chat_history_key].append({"role": "assistant", "content": f"⚠️ Connection Error: {error_msg}"})
                
                st.rerun()

# ----------------- Theme Setup (Strict Dark Mode) -----------------
st.markdown(get_theme_css(), unsafe_allow_html=True)

# Premium Header Section (Strictly Dark-Luxury Glassmorphism)
st.markdown(
    "<div class='glass-header-panel'>"
    "<div class='main-title'>TalentVantage</div>"
    "<div class='subtitle'>Premium Objective & Bias-Mitigated Candidate Screener</div>"
    "</div>",
    unsafe_allow_html=True
)

# Initialize session caches
if 'parsed_uploads' not in st.session_state:
    st.session_state.parsed_uploads = {}
if 'kanban_stages' not in st.session_state:
    st.session_state.kanban_stages = {}
# Chat Window is OPEN by default at startup floating in the right corner
if 'chat_open' not in st.session_state:
    st.session_state.chat_open = True

# Check model assets
if not match_model:
    st.warning("⚠️ Application is running in fallback mode. Please run `python src/model.py` to compile model assets.")
    st.stop()

# ----------------- Main Layout Sections -----------------

# Section 1: Job Description & Upload Resumes (Targeted container styled in CSS)
step1_container = st.container()
with step1_container:
    st.markdown('<div class="step1-card-marker"></div>', unsafe_allow_html=True)
    st.markdown("<h3 style='margin-top:0;'>🎯 Define Opening & Upload Resumes</h3>", unsafe_allow_html=True)
    col_jd_sel, col_jd_txt, col_uploads = st.columns([1, 1.5, 1.5])

    with col_jd_sel:
        jd_preset = st.selectbox("Load Preset Job Opening:", ["Custom Input"] + list(demo_jobs.keys()))
        if jd_preset != "Custom Input":
            preset_info = demo_jobs[jd_preset]
            jd_title = preset_info['title']
            jd_text_val = preset_info['text']
            req_exp_val = preset_info['experience']
        else:
            jd_title = "Custom Job Opening"
            jd_text_val = ""
            req_exp_val = 0

    with col_jd_txt:
        jd_text = st.text_area("Job Description Scope & Requirements:", value=jd_text_val, height=180, placeholder="Paste requirements here...")

    with col_uploads:
        uploaded_files = st.file_uploader(
            "Upload Candidate Resumes (DOCX/PDF/TXT):", 
            type=["docx", "pdf", "txt"], 
            accept_multiple_files=True
        )

# Parse uploaded custom files
raw_candidates = []
if uploaded_files:
    for f in uploaded_files:
        cache_key = f"{f.name}_{f.size}"
        if cache_key in st.session_state.parsed_uploads:
            raw_candidates.append(st.session_state.parsed_uploads[cache_key])
        else:
            temp_dir = os.path.join(BASE_DIR, '../temp_uploads')
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, f.name)
            with open(temp_path, 'wb') as temp_f:
                temp_f.write(f.read())
            
            try:
                ext = f.name.lower()
                if ext.endswith(('.docx', '.pdf', '.txt')):
                    profile = parse_resume(temp_path, skills_vocab)
                    st.session_state.parsed_uploads[cache_key] = profile
                    raw_candidates.append(profile)
            except Exception as e:
                st.error(f"Error parsing file {f.name}: {e}")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

if not jd_text.strip():
    render_chatbot(None, "", "")
    st.info("ℹ️ Please enter or load a Job Description to proceed.")
    st.stop()

if not raw_candidates:
    render_chatbot(None, "", "")
    st.markdown(f"""
    <div style="text-align: center; padding: 60px 20px; background-color: var(--bg-card); border: 2px dashed var(--border); border-radius: 16px; margin-top: 20px; box-shadow: var(--shadow);">
        <div style="font-size: 3.5rem; margin-bottom: 15px;">📁</div>
        <h3 style="margin: 0; color: var(--text-high); font-family: 'Outfit', sans-serif;">Awaiting Resumes</h3>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Section 2: Screening Settings Configuration (Targeted container styled in CSS)
match_engine = "High-End Semantic Match (Recommended)"
step2_container = st.container()
with step2_container:
    st.markdown('<div class="step2-card-marker"></div>', unsafe_allow_html=True)
    st.markdown("<h3 style='margin-top:0;'>⚙️ Configure Match Settings</h3>", unsafe_allow_html=True)
    col_conf_blind, col_conf_score, col_conf_exp = st.columns(3)

    with col_conf_blind:
        blind_screening = st.checkbox("Activate Blind Screening", value=True, help="Hides candidate names, emails, phones, and locations to mitigate bias.")

    with col_conf_score:
        min_match_score = st.slider("Minimum Match Score (%)", min_value=0, max_value=100, value=50)

    with col_conf_exp:
        min_experience = st.slider("Minimum Experience (Years)", min_value=0, max_value=20, value=0)

# Apply Blind Redaction if selected
candidates = []
for idx, profile in enumerate(raw_candidates):
    if blind_screening:
        candidates.append(redact_profile(profile, idx+1))
    else:
        candidates.append(profile)

# Extract JD values dynamically
jd_skills = extract_skills_from_text(jd_text, skills_vocab)
jd_min_exp = parse_required_experience(jd_text) if jd_preset == "Custom Input" else req_exp_val
jd_edu_level = parse_education_level(jd_text)

# Initialize BERT model if chosen
if match_engine.startswith("High-End Semantic"):
    tokenizer, model = load_bert_model()

# Calculate Match Scores for all candidates
scored_candidates = []
for profile in candidates:
    try:
        # Calculate common structural features
        common_skills = list(set(profile['skills']).intersection(set(jd_skills)))
        overlap_count = len(common_skills)
        overlap_ratio = overlap_count / len(jd_skills) if len(jd_skills) > 0 else 0.0
        
        c_exp = profile['experience_years']
        exp_diff = c_exp - jd_min_exp
        
        c_edu = profile['education_level']
        edu_diff = c_edu - jd_edu_level

        # Compute match score based on selected engine
        if match_engine.startswith("High-End Semantic"):
            # Encode JD
            j_combined_text = (jd_title + " " + jd_text)
            j_emb = get_bert_embedding(j_combined_text, tokenizer, model)
            
            # Retrieve cached candidate embedding or encode on-the-fly
            c_emb = bert_embeddings.get(profile['filename'])
            if c_emb is None:
                c_combined_text = (
                    f"Name: {profile['name']}\n"
                    f"Education: {profile['education_degree']}\n"
                    f"Experience: {profile['experience_years']} years\n"
                    f"Skills: {', '.join(profile['skills'])}\n"
                    f"Content: {profile['text']}"
                )
                c_emb = get_bert_embedding(c_combined_text, tokenizer, model)
                
            # Compute Cosine similarity
            norm_c = np.linalg.norm(c_emb)
            norm_j = np.linalg.norm(j_emb)
            sim = np.dot(c_emb, j_emb) / (norm_c * norm_j) if (norm_c > 0 and norm_j > 0) else 0.0
            
            # Scale similarity [0.2, 0.8] -> [0, 100]%
            final_score = int(np.clip((sim - 0.2) / 0.6 * 100, 0, 100))
        else:
            # TF-IDF Cosine Similarity
            c_combined_text = (profile['name'] + " " + " ".join(profile['skills']) + " " + profile['text'])
            j_combined_text = (jd_title + " " + jd_text)
            
            c_tfidf = tfidf_vectorizer.transform([c_combined_text])
            j_tfidf = tfidf_vectorizer.transform([j_combined_text])
            sim = cosine_similarity(c_tfidf, j_tfidf)[0][0]
            if np.isnan(sim):
                sim = 0.0
                
            features = np.array([[sim, overlap_count, overlap_ratio, exp_diff, edu_diff]])
            predicted_val = match_model.predict(features)[0]
            final_score = int(np.clip(predicted_val * 100, 0, 100))

        scored_candidates.append({
            'profile': profile,
            'match_score': final_score,
            'sim': sim,
            'common_skills': common_skills,
            'missing_skills': sorted(list(set(jd_skills) - set(profile['skills']))),
            'exp_diff': exp_diff,
            'edu_diff': edu_diff,
            'exp_years': c_exp,
            'edu_degree': profile['education_degree']
        })
    except Exception as e:
        print(f"Error scoring {profile['filename']}: {e}")

# Sort candidates by score
scored_candidates.sort(key=lambda x: x['match_score'], reverse=True)

# Filter
filtered_candidates = [
    c for c in scored_candidates 
    if c['match_score'] >= min_match_score and c['exp_years'] >= min_experience
]

# Initialize Kanban Stages in session state
for c in scored_candidates:
    if c['profile']['filename'] not in st.session_state.kanban_stages:
        st.session_state.kanban_stages[c['profile']['filename']] = 'Applied'

# Section 3: Overview Cards
st.subheader("📊 Screening Overview")
card1, card2, card3, card4 = st.columns(4)

with card1:
    st.markdown(f"<div class='metric-card'><div class='metric-val'>{len(candidates)}</div><div class='metric-lbl'>Resumes Scanned</div></div>", unsafe_allow_html=True)
with card2:
    st.markdown(f"<div class='metric-card'><div class='metric-val'>{len(filtered_candidates)}</div><div class='metric-lbl'>Passed Filter</div></div>", unsafe_allow_html=True)
with card3:
    max_score = scored_candidates[0]['match_score'] if scored_candidates else 0
    st.markdown(f"<div class='metric-card'><div class='metric-val'>{max_score}%</div><div class='metric-lbl'>Top Match Score</div></div>", unsafe_allow_html=True)
with card4:
    avg_score = int(np.mean([c['match_score'] for c in scored_candidates])) if scored_candidates else 0
    st.markdown(f"<div class='metric-card'><div class='metric-val'>{avg_score}%</div><div class='metric-lbl'>Average Score</div></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Section 4: SPA Tabs (Candidate Rankings, Kanban, Clustering)
tab_rankings, tab_kanban, tab_clustering = st.tabs([
    "🏆 Candidate Rankings", 
    "📋 Recruitment Kanban Board", 
    "🔮 AI Clustering"
])

# -- Tab 1: Rankings & Inspector --
with tab_rankings:
    if not filtered_candidates:
        st.info("No candidates match your current filter settings in the configuration panel.")
    else:
        st.subheader("Ranked Applicants")
        
        table_data = []
        for idx, item in enumerate(filtered_candidates):
            p = item['profile']
            table_data.append({
                "Rank": idx + 1,
                "Name": p['name'],
                "Match Score": f"{item['match_score']}%",
                "Years Exp": f"{item['exp_years']} yrs",
                "Degree": item['edu_degree'],
                "Email": p['email'],
                "Location": p['location'],
                "Skills Match Count": f"{len(item['common_skills'])} / {len(jd_skills)}",
                "id": idx
            })
            
        df_table = pd.DataFrame(table_data)
        st.dataframe(df_table.drop(columns=['id']), use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.subheader("🔍 Deep Applicant Inspector")
        
        inspect_options = [f"Rank #{item['Rank']}: {item['Name']} (Match: {item['Match Score']})" for item in table_data]
        selected_option = st.selectbox("Select Candidate to Inspect:", inspect_options, key="rank_select")
        selected_idx = inspect_options.index(selected_option)
        c_detail = filtered_candidates[selected_idx]
        p_detail = c_detail['profile']
        
        col_score, col_details = st.columns([1, 2])
        
        with col_score:
            # Match Gauge
            try:
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge",
                    value = c_detail['match_score'],
                    domain = {'x': [0, 1], 'y': [0.2, 1]},
                    title = {'text': "Match Compatibility", 'font': {'size': 18, 'family': 'Outfit', 'color': '#f3f4f6'}},
                    gauge = {
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#8b949e"},
                        'bar': {'color': "#818cf8"},
                        'bgcolor': "rgba(128,128,128,0.1)",
                        'borderwidth': 2,
                        'bordercolor': "#30363d",
                        'steps': [
                            {'range': [0, 50], 'color': 'rgba(239, 68, 68, 0.1)'},
                            {'range': [50, 75], 'color': 'rgba(192, 132, 252, 0.1)'},
                            {'range': [75, 100], 'color': 'rgba(16, 185, 129, 0.1)'}
                        ]
                    }
                ))
                fig_gauge.add_annotation(
                    text=f"<span style='font-size:42px; font-weight:800; color:#f3f4f6; font-family:Outfit;'>{c_detail['match_score']}%</span>",
                    x=0.5,
                    y=0.15,
                    showarrow=False
                )
                fig_gauge.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font={'family': "Outfit"},
                    height=250,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig_gauge, use_container_width=True)
            except Exception as ex:
                st.write(f"Could not render Gauge chart: {ex}")
            
            # Contact details box
            st.markdown(f"""
            <div style="background-color: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; padding: 15px;">
                <h4 style="margin-top:0; font-family: Outfit;">Contact Info</h4>
                <p><b>Email:</b> {p_detail['email']}</p>
                <p><b>Phone:</b> {p_detail['phone']}</p>
                <p><b>Location:</b> {p_detail['location']}</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col_details:
            detail_tabs = st.tabs(["🎯 Skill Gap Analysis", "🤖 Explainable AI Insights", "📋 ATS Check & Heuristics", "🛡️ Bias Flag Check", "📄 Highlighted Resume Text"])
            
            # Tab 1: Skill Gap
            with detail_tabs[0]:
                st.write("**Candidate Skills Comparison**")
                matched = c_detail['common_skills']
                missing = c_detail['missing_skills']
                
                st.write("**Matching Skills (Green):**")
                if matched:
                    pills_html = "".join([f"<span class='tag-match'>{s}</span>" for s in matched])
                    st.markdown(pills_html, unsafe_allow_html=True)
                else:
                    st.write("None")
                    
                st.write("<br>**Missing Required Skills (Red):**", unsafe_allow_html=True)
                if missing:
                    pills_html = "".join([f"<span class='tag-gap'>{s}</span>" for s in missing])
                    st.markdown(pills_html, unsafe_allow_html=True)
                else:
                    st.write("No missing skills identified! Perfect alignment.")
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Radar Chart
                try:
                    categories = ['Semantic Sim', 'Skill Overlap', 'Experience Align', 'Education Align']
                    candidate_vals = [
                        c_detail['sim'],
                        len(c_detail['common_skills'])/max(1, len(jd_skills)),
                        1.0 if c_detail['exp_diff'] >= 0 else max(0.0, 1.0 + c_detail['exp_diff']/5.0),
                        1.0 if c_detail['edu_diff'] >= 0 else 0.5
                    ]
                    
                    fig_radar = go.Figure()
                    fig_radar.add_trace(go.Scatterpolar(
                        r=candidate_vals,
                        theta=categories,
                        fill='toself',
                        name='Candidate Profile Fit',
                        line_color='#c084fc',
                        fillcolor='rgba(192, 132, 252, 0.2)'
                    ))
                    fig_radar.update_layout(
                        polar=dict(
                            radialaxis=dict(visible=True, range=[0, 1]),
                            bgcolor='rgba(128,128,128,0.05)'
                        ),
                        paper_bgcolor='rgba(0,0,0,0)',
                        height=280,
                        margin=dict(l=40, r=40, t=20, b=20)
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)
                except Exception as ex:
                    st.write(f"Could not render radar chart: {ex}")
                
                if missing:
                    st.info(f"💡 **Suggested Upskilling Path:** The recruiter recommends candidate acquires **{', '.join(missing[:3])}** to maximize performance in this role.")
                
            # Tab 2: Explainable AI
            with detail_tabs[1]:
                st.write("**AI Semantic Matching Insights**")
                st.write(f"""
                The compatibility score is calculated using advanced semantic embeddings mapping both the candidate's profile and the job description requirements.
                - **Semantic Similarity Coefficient**: `{c_detail['sim']:.4f}`
                - Unlike classical keyword matching, this model captures context-aware phrases, synonyms, and related experience mappings.
                """)
                
                # Component Breakdown Chart
                contrib_data = pd.DataFrame({
                    "Component": ["Semantic Similarity Coefficient", "Skills Overlap Ratio", "Experience Align", "Education Align"],
                    "Value (%)": [
                        int(c_detail['sim'] * 100),
                        int((len(matched)/max(1, len(jd_skills))) * 100),
                        100 if c_detail['exp_diff'] >= 0 else int(max(0, 1.0 + c_detail['exp_diff']/5.0)*100),
                        100 if c_detail['edu_diff'] >= 0 else 50
                    ]
                })
                fig_contrib = px.bar(
                    contrib_data, 
                    x="Value (%)", 
                    y="Component", 
                    orientation='h',
                    color_discrete_sequence=['#818cf8'],
                    text="Value (%)"
                )
                fig_contrib.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(128, 128, 128, 0.05)',
                    height=200,
                    margin=dict(l=10, r=10, t=10, b=10),
                    xaxis=dict(showgrid=False),
                    yaxis=dict(autorange="reversed")
                )
                st.plotly_chart(fig_contrib, use_container_width=True)
                
                st.write(f"""
                **AI Assessment Summary:**
                - **General Compatibility:** Candidate displays a **{int(c_detail['sim']*100)}%** general semantic similarity with the responsibilities.
                - **Skill Coverage:** The applicant possesses **{len(matched)} out of {len(jd_skills)}** skills explicitly requested by the job description.
                - **Experience Fit:** Candidate possesses **{c_detail['exp_years']} years** of experience. The role requires **{jd_min_exp} years** ({"Match/Exceeds" if c_detail['exp_diff'] >= 0 else f"Short by {abs(c_detail['exp_diff'])} years"}).
                - **Education Match:** Candidate holds a **{c_detail['edu_degree']}**, while the role asks for a level corresponding to **{demo_jobs[jd_preset]['education'] if jd_preset != 'Custom Input' else 'Bachelor'}** level.
                """)
                
            # Tab 3: ATS Check
            with detail_tabs[2]:
                st.markdown("### 📊 Advanced Recruiter-Grade ATS Scorecard")
                st.write("This professional evaluation scores candidate credentials, layout indexability, and content impact on a weighted 100-point scale.")
                
                # Setup categories and components
                word_count = len(p_detail['text'].split())
                has_email = p_detail['email'] != "N/A" and "@" in p_detail['email']
                has_phone = p_detail['phone'] != "N/A" and any(char.isdigit() for char in p_detail['phone'])
                
                # 1. Core Skills Alignment (30 pts max)
                SOFT_SKILLS_SET = {
                    'agile', 'scrum', 'kanban', 'project management', 'program management', 'product management',
                    'leadership', 'communication', 'teamwork', 'problem-solving', 'collaboration', 'critical thinking',
                    'time management', 'adaptability', 'creativity', 'scrum master', 'scrummaster', 'devops',
                    'sdlc', 'waterfall', 'presentation', 'mentoring', 'negotiation', 'analytical'
                }
                
                jd_tech_skills = [s for s in jd_skills if s.lower() not in SOFT_SKILLS_SET]
                jd_soft_skills = [s for s in jd_skills if s.lower() in SOFT_SKILLS_SET]
                
                cand_tech_skills = [s for s in c_detail['common_skills'] if s.lower() not in SOFT_SKILLS_SET]
                cand_soft_skills = [s for s in c_detail['common_skills'] if s.lower() in SOFT_SKILLS_SET]
                
                # Tech Match (20 pts)
                tech_total = len(jd_tech_skills)
                tech_match = len(cand_tech_skills)
                tech_ratio = tech_match / tech_total if tech_total > 0 else 1.0
                if tech_ratio >= 0.75:
                    score_tech = 20
                    desc_tech = f"**Excellent Technical Fit** ({tech_match}/{tech_total} matching). The candidate possesses the majority of required hard skills and technical requirements."
                elif tech_ratio >= 0.40:
                    score_tech = 14
                    desc_tech = f"**Moderate Technical Fit** ({tech_match}/{tech_total} matching). Candidate has some core technical skills but is missing key stack components."
                elif tech_ratio >= 0.15:
                    score_tech = 6
                    desc_tech = f"**Low Technical Fit** ({tech_match}/{tech_total} matching). Significant gaps identified in the required technical stacks."
                else:
                    score_tech = 0
                    desc_tech = f"**Poor Technical Fit** ({tech_match}/{tech_total} matching). Almost no technical skills overlap with job description."
                    
                # Soft Match (10 pts)
                soft_total = len(jd_soft_skills)
                soft_match = len(cand_soft_skills)
                soft_ratio = soft_match / soft_total if soft_total > 0 else 1.0
                if soft_ratio >= 0.75:
                    score_soft = 10
                    desc_soft = f"**Strong Process Alignment** ({soft_match}/{soft_total} matching). Meets agile, methodology, and soft skills targets."
                elif soft_ratio >= 0.40:
                    score_soft = 7
                    desc_soft = f"**Moderate Process Alignment** ({soft_match}/{soft_total} matching). Meets some process requirements."
                elif soft_ratio >= 0.15:
                    score_soft = 3
                    desc_soft = f"**Low Process Alignment** ({soft_match}/{soft_total} matching). Significant soft skills or methodology gaps."
                else:
                    score_soft = 0
                    desc_soft = f"**No Process Alignment** ({soft_match}/{soft_total} matching). Missing required soft skills or project methodologies."
                    
                # 2. Role Fit & Career Progression (25 pts max)
                # Experience Alignment (15 pts)
                exp_diff = c_detail['exp_diff']
                if exp_diff >= 0:
                    score_exp = 15
                    desc_exp = f"**Experience Target Met**. Candidate has {c_detail['exp_years']} years of experience, meeting or exceeding the required {jd_min_exp} years (excess of {exp_diff:.1f} yrs)."
                elif exp_diff >= -1.5:
                    score_exp = 9
                    desc_exp = f"**Experience Slightly Short**. Candidate has {c_detail['exp_years']} years of experience, slightly short of target {jd_min_exp} years (deficit of {abs(exp_diff):.1f} yrs)."
                else:
                    score_exp = 3
                    desc_exp = f"**Experience Deficit**. Candidate has only {c_detail['exp_years']} years of experience, significantly short of target {jd_min_exp} years (deficit of {abs(exp_diff):.1f} yrs)."
                    
                # Career Growth (5 pts)
                progression_keywords = ['lead', 'senior', 'principal', 'promoted', 'spearheaded', 'pioneered', 'architected', 'manager', 'director']
                found_progression = [kw for kw in progression_keywords if kw in p_detail['text'].lower()]
                found_progression = list(set(found_progression))
                if len(found_progression) >= 2:
                    score_prog = 5
                    desc_prog = f"**Career Growth Indicators Found**. Resume text contains keywords indicating leadership, promotion, or ownership: {', '.join(found_progression[:3])}."
                else:
                    score_prog = 2
                    desc_prog = "**No clear indicators of progression/leadership keywords** found in resume text."
                    
                # Stability Check (5 pts)
                years = re.findall(r'\b(20\d{2})\b', p_detail['text'])
                duplicate_years = len(years) - len(set(years))
                if duplicate_years > 8:
                    score_stability = 3
                    desc_stability = "**Stability Warning**: Frequent date shifts detected in text. Recommend auditing for short job durations."
                else:
                    score_stability = 5
                    desc_stability = "**Stability Verified**. Dates represent consistent and standard tenure profiles."
                    
                # 3. Education & Credentials Match (15 pts max)
                # Education Fit (10 pts)
                edu_diff = c_detail['edu_diff']
                if edu_diff >= 0:
                    score_edu = 10
                    desc_edu = f"**Degree Level Met/Exceeded**. Candidate has {p_detail['education_degree']} (matches or exceeds required level)."
                elif edu_diff == -1:
                    score_edu = 5
                    desc_edu = f"**Degree Slightly Below Target**. Candidate has {p_detail['education_degree']}, but target requires 1 higher qualification level."
                else:
                    score_edu = 1
                    desc_edu = f"**Degree Deficit**. Candidate has {p_detail['education_degree']} (target is significantly higher)."
                    
                # Certifications (5 pts)
                certifications_keywords = ['pmp', 'aws certified', 'csm', 'cissp', 'oracle', 'certified scrum', 'ccna', 'ccnp', 'itil', 'prince2', 'gcp', 'azure certified']
                found_certs = [cert for cert in certifications_keywords if cert in p_detail['text'].lower()]
                found_certs = list(set(found_certs))
                if found_certs:
                    score_certs = 5
                    desc_certs = f"**Professional Credentials Found**. Identified industry certification keywords: {', '.join(found_certs)}."
                else:
                    score_certs = 2
                    desc_certs = "**No industry certifications identified** in resume text."
                    
                # 4. Layout Readability & ATS Indexability (15 pts max)
                # Layout Structure (10 pts)
                headers = ['education', 'experience', 'skills', 'objective', 'projects', 'summary', 'languages', 'certifications']
                found_headers = [h for h in headers if h in p_detail['text'].lower()]
                found_headers = list(set(found_headers))
                if len(found_headers) >= 4:
                    score_headers = 10
                    desc_headers = f"**Strong Sectioning Found**. ATS can read standard layout zones. Headers found: {', '.join(found_headers[:5])}."
                elif len(found_headers) >= 2:
                    score_headers = 5
                    desc_headers = f"**Partial Sectioning**. Standardize headers (e.g. use 'Education', 'Work History') to improve ATS parser parsing accuracy."
                else:
                    score_headers = 0
                    desc_headers = "**Poor Layout Structure**. Missing key standard section headers. Text blocks may fail to index correctly."
                    
                # Paragraph Density / formatting warnings (5 pts)
                paragraphs = [p.strip() for p in p_detail['text'].split('\n') if len(p.strip()) > 0]
                overly_dense = [p for p in paragraphs if len(p.split()) > 150]
                if overly_dense:
                    score_density = 2
                    desc_density = f"**Formatting Warning**: Detected {len(overly_dense)} blocks of dense text exceeding 150 words. Large block paragraphs reduce parser readability."
                else:
                    score_density = 5
                    desc_density = "**Formatting Spacing Passed**. Document text is well-sectioned and uses digestible paragraphs."
                    
                # 5. Content Quality & Impact (15 pts max)
                # Action Verbs (5 pts)
                action_verbs = ['spearheaded', 'designed', 'implemented', 'optimized', 'engineered', 'pioneered', 'led', 'built', 'managed', 'architected', 'conducted', 'developed', 'collaborated', 'created', 'streamlined', 'increased', 'reduced']
                found_verbs = [v for v in action_verbs if rf'\b{v}' in p_detail['text'].lower()]
                found_verbs = list(set(found_verbs))
                if len(found_verbs) >= 5:
                    score_verbs = 5
                    desc_verbs = f"**High Impact Verbs Used**. Found strong professional action-oriented vocabulary: {', '.join(found_verbs[:5])}."
                else:
                    score_verbs = 2
                    desc_verbs = f"**Passive Voice / Low Impact Verbs**. Found only {len(found_verbs)} robust action verbs. Increase verbs like 'spearheaded' to sound more impact-driven."
                    
                # Quantifiable Metrics (5 pts)
                metrics_matches = re.findall(r'\b\d+(?:\.\d+)?%|\$\d+(?:\.\d+)?\s*(?:million|k|billion)?|\b\d+\+\s*users\b|\b\d+\s*fold\b', p_detail['text'])
                metrics_matches = list(set(metrics_matches))
                if len(metrics_matches) >= 2:
                    score_metrics = 5
                    desc_metrics = f"**Quantified Results Found**. Document contains numerical business metrics showing achievements: {', '.join(metrics_matches[:3])}."
                else:
                    score_metrics = 1
                    desc_metrics = "**No Quantifiable Metrics**. Recruiters prefer achievements backstopped with figures (e.g. 'boosted efficiency by 15%')."
                    
                # Word Count (5 pts)
                if 400 <= word_count <= 1000:
                    score_words = 5
                    desc_words = f"**Optimal Length** ({word_count} words). Ideal size for modern single-page or double-page indexation."
                else:
                    score_words = 2
                    desc_words = f"**Suboptimal Length** ({word_count} words). Resume is too short or wordy. Aim for 400 - 1000 words."
                    
                # Calculate final score
                ats_score = (score_tech + score_soft + score_exp + score_prog + score_stability + 
                             score_edu + score_certs + score_headers + score_density + 
                             score_verbs + score_metrics + score_words)
                
                # Render ATS Dashboard Gauge
                col_m1, col_m2 = st.columns([1, 2])
                with col_m1:
                    st.metric("Advanced Recruiter-Grade Score", f"{ats_score}/100")
                with col_m2:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.progress(ats_score / 100.0)
                
                st.markdown("---")
                st.write("**ATS Scorecard Details**")
                
                # Render Accordion Expanders
                with st.expander(f"🎯 Core Skills Alignment (Score: {score_tech + score_soft}/30)"):
                    st.write(f"**Technical/Hard Skills Match ({score_tech}/20)**")
                    st.write(f"<span style='color:#8b949e; font-size:0.85rem;'>{desc_tech}</span>", unsafe_allow_html=True)
                    st.write(f"**Soft/Methodology Skills Match ({score_soft}/10)**")
                    st.write(f"<span style='color:#8b949e; font-size:0.85rem;'>{desc_soft}</span>", unsafe_allow_html=True)
                    
                with st.expander(f"📈 Experience & Role Seniority (Score: {score_exp + score_prog + score_stability}/25)"):
                    st.write(f"**Experience Compatibility ({score_exp}/15)**")
                    st.write(f"<span style='color:#8b949e; font-size:0.85rem;'>{desc_exp}</span>", unsafe_allow_html=True)
                    st.write(f"**Career Progression Heuristics ({score_prog}/5)**")
                    st.write(f"<span style='color:#8b949e; font-size:0.85rem;'>{desc_prog}</span>", unsafe_allow_html=True)
                    st.write(f"**Tenure & Stability Check ({score_stability}/5)**")
                    st.write(f"<span style='color:#8b949e; font-size:0.85rem;'>{desc_stability}</span>", unsafe_allow_html=True)
                    
                with st.expander(f"🎓 Education & Credentials Fit (Score: {score_edu + score_certs}/15)"):
                    st.write(f"**Degree Qualification Match ({score_edu}/10)**")
                    st.write(f"<span style='color:#8b949e; font-size:0.85rem;'>{desc_edu}</span>", unsafe_allow_html=True)
                    st.write(f"**Industry Credentials & Certifications ({score_certs}/5)**")
                    st.write(f"<span style='color:#8b949e; font-size:0.85rem;'>{desc_certs}</span>", unsafe_allow_html=True)
                    
                with st.expander(f"📋 Layout Indexability & Formatting (Score: {score_headers + score_density}/15)"):
                    st.write(f"**ATS Sectioning and Headers ({score_headers}/10)**")
                    st.write(f"<span style='color:#8b949e; font-size:0.85rem;'>{desc_headers}</span>", unsafe_allow_html=True)
                    st.write(f"**Formatting Spacing & Text Density ({score_density}/5)**")
                    st.write(f"<span style='color:#8b949e; font-size:0.85rem;'>{desc_density}</span>", unsafe_allow_html=True)
                    
                with st.expander(f"✍️ Action Quality & Quantifiable Impact (Score: {score_verbs + score_metrics + score_words}/15)"):
                    st.write(f"**Impact-Driven Action Verbs ({score_verbs}/5)**")
                    st.write(f"<span style='color:#8b949e; font-size:0.85rem;'>{desc_verbs}</span>", unsafe_allow_html=True)
                    st.write(f"**Measurable & Quantifiable Results ({score_metrics}/5)**")
                    st.write(f"<span style='color:#8b949e; font-size:0.85rem;'>{desc_metrics}</span>", unsafe_allow_html=True)
                    st.write(f"**Optimal Word Density ({score_words}/5)**")
                    st.write(f"<span style='color:#8b949e; font-size:0.85rem;'>{desc_words}</span>", unsafe_allow_html=True)
                
                # Recommendations list
                st.markdown("<br>**💡 Recruiter Recommendations for Resume Optimization**", unsafe_allow_html=True)
                recommendations = []
                if score_density < 5:
                    recommendations.append("Break up overly dense text blocks (exceeding 150 words) into clean bullet points with a focus on single key metrics.")
                if score_verbs < 5:
                    recommendations.append("Strengthen descriptions with professional action-oriented verbs (e.g. *spearheaded*, *engineered*, *architected*) rather than passive descriptions.")
                if score_metrics < 5:
                    recommendations.append("Quantify achievements by including concrete figures and business impact indicators (e.g., percentages, financial goals, scale of users).")
                if score_headers < 10:
                    recommendations.append("Standardize section header titles (e.g. 'Professional Experience', 'Education') to assist automated ATS parsers.")
                if score_certs < 5:
                    recommendations.append("Acquire and list relevant professional credentials/certifications (e.g. PMP, CSM, AWS/Azure certifications) to demonstrate continuous learning.")
                
                if recommendations:
                    for rec in recommendations:
                        st.markdown(f"- <span style='font-size:0.85rem; color:#818cf8;'>{rec}</span>", unsafe_allow_html=True)
                else:
                    st.markdown("- <span style='font-size:0.85rem; color:#10b981;'>No issues found! Your resume formatting and content structure are highly optimized for automated scanning.</span>", unsafe_allow_html=True)
                    
            # Tab 4: Bias Check
            with detail_tabs[3]:
                st.write("**Job Description Diversity Check**")
                bias_results = analyze_bias(jd_text)
                
                col_b1, col_b2 = st.columns(2)
                with col_b1:
                    st.write("**Job Description Gender Balance:**")
                    balance_color = "#10b981" if bias_results['gender_balance'] == "Neutral" else "#ef4444"
                    st.markdown(f"<h4 style='color:{balance_color};'>{bias_results['gender_balance']}</h4>", unsafe_allow_html=True)
                    
                with col_b2:
                    st.write("**Bias Flag Indicators:**")
                    st.write(f"- Masculine coded terms: {len(bias_results['jd_masculine'])}")
                    st.write(f"- Feminine coded terms: {len(bias_results['jd_feminine'])}")
                    st.write(f"- Age-discriminatory terms: {len(bias_results['jd_age'])}")
                
                st.markdown("<br>", unsafe_allow_html=True)
                if bias_results['jd_masculine'] or bias_results['jd_feminine'] or bias_results['jd_age']:
                    st.write("**Identified Terms in Job Description:**")
                    if bias_results['jd_masculine']:
                        st.write(f"Masculine coded: {', '.join([f'`{w}`' for w in bias_results['jd_masculine']])}")
                    if bias_results['jd_feminine']:
                        st.write(f"Feminine coded: {', '.join([f'`{w}`' for w in bias_results['jd_feminine']])}")
                    if bias_results['jd_age']:
                        st.warning(f"⚠️ Age-related triggers: {', '.join([f'`{w}`' for w in bias_results['jd_age']])} (Consider reframing to encourage older/younger applicants equally).")
                else:
                    st.write("No gender-coded or age-discriminatory language flagged in the Job Description!")
                    
            # Tab 5: Highlighted Resume Text
            with detail_tabs[4]:
                st.write("Overlapping skills are highlighted in **green**; warnings / career gaps are highlighted in **red**.")
                hl_html = highlight_resume_text(p_detail['text'], matched)
                st.markdown(f"""
                <div class="resume-viewer">
                {hl_html}
                </div>
                """, unsafe_allow_html=True)

# -- Tab 2: Kanban Pipeline Board --
with tab_kanban:
    st.subheader("📋 Recruitment Kanban Board")
    
    stages = ["Applied", "Screened", "Interview Scheduled", "Offered"]
    cols = st.columns(4)
    
    for col_idx, stage in enumerate(stages):
        with cols[col_idx]:
            stage_c = [c for c in scored_candidates if st.session_state.kanban_stages.get(c['profile']['filename'], 'Applied') == stage]
            
            st.markdown(f"""
            <div class='kanban-header'>
                {stage} ({len(stage_c)})
            </div>
            """, unsafe_allow_html=True)
            
            for item in stage_c:
                p = item['profile']
                st.markdown(f"""
                <div class='kanban-card'>
                    <h5 style='margin:0 0 5px 0; color:#818cf8; font-family: Outfit;'>{p['name']}</h5>
                    <p style='margin:0 0 3px 0; font-size:0.85rem;'>Score: <b>{item['match_score']}%</b></p>
                    <p style='margin:0 0 8px 0; font-size:0.8rem; color:#9ca3af;'>Exp: {item['exp_years']} yrs | {item['edu_degree']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                new_stage = st.selectbox(
                    "Move Stage:", 
                    stages, 
                    index=stages.index(stage),
                    key=f"kb_{p['filepath']}",
                    label_visibility="collapsed"
                )
                if new_stage != stage:
                    st.session_state.kanban_stages[p['filename']] = new_stage
                    st.rerun()

# -- Tab 3: Clustering --
with tab_clustering:
    st.subheader("🔮 Candidate Profile Groupings (Unsupervised Clustering)")
    
    try:
        c_texts = [(p['name'] + " " + " ".join(p['skills']) + " " + p['text']) for p in candidates]
        c_tfidf_matrix = tfidf_vectorizer.transform(c_texts)
        
        num_clusters = min(4, len(candidates))
        if num_clusters >= 2:
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
            clusters = kmeans.fit_predict(c_tfidf_matrix)
            
            pca = PCA(n_components=2, random_state=42)
            pca_coords = pca.fit_transform(c_tfidf_matrix.toarray())
            
            plot_df = pd.DataFrame({
                "x": pca_coords[:, 0],
                "y": pca_coords[:, 1],
                "Candidate": [p['name'] for p in candidates],
                "Cluster": [f"Cluster {c+1}" for c in clusters],
                "Top Skills": [", ".join(p['skills'][:4]) for p in candidates]
            })
            
            fig_scatter = px.scatter(
                plot_df, 
                x="x", 
                y="y", 
                color="Cluster",
                hover_data=["Candidate", "Top Skills"],
                title="PCA Projection of Candidate Profiles",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_scatter.update_layout(
                paper_bgcolor='rgba(128,128,128,0.02)',
                plot_bgcolor='rgba(128,128,128,0.02)',
                font={'color': '#cbd5e1'},
                height=500,
                xaxis=dict(showgrid=False, zeroline=False, title="PCA Component 1"),
                yaxis=dict(showgrid=False, zeroline=False, title="PCA Component 2")
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            col_pools = st.columns(num_clusters)
            for c_idx in range(num_clusters):
                with col_pools[c_idx]:
                    st.markdown(f"**Group Pool {c_idx+1}**")
                    grp_candidates = plot_df[plot_df["Cluster"] == f"Cluster {c_idx+1}"]["Candidate"].tolist()
                    st.write(f"<span style='color:#8b949e; font-size:0.85rem;'>{len(grp_candidates)} candidates grouped</span>", unsafe_allow_html=True)
                    for name in grp_candidates[:8]:
                        st.markdown(f"- {name}")
                    if len(grp_candidates) > 8:
                        st.markdown(f"*...and {len(grp_candidates)-8} more*")
        else:
            st.info("ℹ️ Need at least 2 candidates to cluster profiles.")
    except Exception as ex:
        st.write(f"Clustering error: {ex}")

# Render Chatbot at the end of the page for Path B (dashboard fully loaded)
render_chatbot(scored_candidates, jd_title, jd_text)