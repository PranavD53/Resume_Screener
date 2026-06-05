# TalentVantage: Premium Objective & Bias-Mitigated Candidate Screener

TalentVantage is a premium, minimalist Single Page Application (SPA) designed to automate resume screening, calculate objective semantic match scores on-the-fly, identify skill gaps, and audit formatting compatibility—all while mitigating unconscious hiring bias.

---

## 💎 Core Features

### 1. High-End Semantic Matching Engine
- **BERT-Based Embeddings**: Translates applicant resumes and job descriptions into high-dimensional semantic vectors using the local cached `sentence-transformers/all-mpnet-base-v2` encoder model.
- **Cosine Similarity Scoring**: Computes objective candidate-to-JD match scores on-the-fly, bypassing the limitations of simple keyword matching and matching criteria based on deep contextual semantics.

### 2. Bias-Mitigated blind Screening
- **Demographic Redaction**: Recruiters can toggle "Blind Screening" to instantly redact applicant names, emails, phone numbers, and location descriptions from the dashboard view.
- **Objective Grading**: Promotes merit-based selection by focusing the screening process entirely on skill matches, years of experience, and educational credentials.

### 3. Deep Applicant Inspector Tabs
- **🎯 Skill Gap Analysis**: Categorizes candidate skills into matching (green) and missing (red) pills relative to job requirements.
- **🤖 Explainable AI Insights**: Visualizes how features (TF-IDF similarity, skills overlap, experience, and education alignment) contributed to the final match score.
- **📋 ATS Compatibility Matrix**: A strict, automated 5-category audit (word length, contact details, standard header layout, skill density threshold, and experience alignment) scoring compatibility out of 100 with recommendations.
- **🛡️ Bias Flag Check**: Analyzes the job description text for demographic bias indicators (gender-coded keywords or age-related triggers).
- **📄 Highlighted Resume Text**: Renders parsed resume text with inline highlights (matching skills in green; warning alerts like career gaps or terminations in red).

### 4. Interactive Floating Co-Pilot
- **Integrated Chat Window**: A floating conversation box that tracks candidate contexts based on the recruiter's active selection.
- **Ollama LLM Integration**: Connects to the local running Ollama instance (prioritizing fast models like `llama3.2`) to answer questions, generate 3-bullet summaries, or draft customized interview questions.
- **Boilerplate-Free Streaming**: Prompt-engineered to eliminate conversational filler and cap generated tokens for rapid 3-4 second execution.

---

## 🛠️ Technology Stack

- **UI & Layout**: Streamlit, HTML5, Vanilla CSS3 (luxury dark-mode glassmorphism and block-scrolling flow)
- **Natural Language Processing**: HuggingFace Transformers, PyTorch CPU (thread-optimized)
- **Generative AI Agent**: Ollama Local API
- **File Parsing**: python-docx, PyPDF2
- **Data & Analytics**: Scikit-Learn, Pandas, NumPy, Plotly

---

## 📁 Repository Structure

```
├── models/                     # Serialized ML models and vocabulary
│   ├── skills_vocab.pkl        # List of compiled standard skills
│   ├── tfidf_vectorizer.pkl    # Pre-fitted TF-IDF vectorizer
│   ├── match_model.pkl         # Trained RandomForestRegressor
│   └── metrics.pkl             # Model performance statistics
├── src/
│   ├── parser.py               # Resume parsing & extraction logic
│   ├── model.py                # Pre-computation model pipeline script
│   └── app.py                  # Streamlit dashboard application
├── Resumes/                    # Input folder for DOCX resumes
├── requirements.txt            # Python dependencies
├── verify_system.py            # Integration test script
├── .gitignore                  # Git exclusions file
└── README.md                   # Project details
```