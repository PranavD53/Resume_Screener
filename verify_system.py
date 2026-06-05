import os
import pickle
import requests
import numpy as np
from src.parser import parse_resume, load_skills_vocab
from sklearn.metrics.pairwise import cosine_similarity

def run_tests():
    print("Running integration tests for Resume Screening System...")
    
    # 1. Verify models exist
    model_dir = 'models'
    assets = ['skills_vocab.pkl', 'tfidf_vectorizer.pkl', 'match_model.pkl', 'metrics.pkl']
    print("\nChecking serialized assets in 'models/':")
    for asset in assets:
        path = os.path.join(model_dir, asset)
        exists = os.path.exists(path)
        print(f" - {asset}: {'FOUND' if exists else 'MISSING'}")
        assert exists, f"Required model asset '{asset}' is missing!"
        
    # 2. Test loading assets
    print("\nLoading assets...")
    skills_vocab = load_skills_vocab()
    with open('models/tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    with open('models/match_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print(f"Successfully loaded {len(skills_vocab)} skills and model components.")
    
    # 3. Parse sample resume
    sample_resume = 'Resumes/Abiral_Pandey_Fullstack_Java.docx'
    print(f"\nParsing sample resume: '{sample_resume}'...")
    profile = parse_resume(sample_resume, skills_vocab)
    
    print("Extracted Profile:")
    print(f" - Name: {profile['name']}")
    print(f" - Email: {profile['email']}")
    print(f" - Phone: {profile['phone']}")
    print(f" - Experience: {profile['experience_years']} years")
    print(f" - Education Level: {profile['education_degree']} (level {profile['education_level']})")
    print(f" - Skills count: {len(profile['skills'])}")
    
    assert profile['name'] == "Abiral Pandey", "Failed to extract correct name!"
    assert profile['email'] == "abiral.pandey88@gmail.com", "Failed to extract correct email!"
    assert len(profile['skills']) > 0, "Failed to extract skills!"
    
    # 4. Score calculation simulation
    print("\nCalculating match score against a sample Senior Java role...")
    jd_title = "Senior Java Developer"
    jd_text = "Looking for a Senior Java Developer with 5+ years experience. Must know Spring Boot, SQL, AWS, and Git."
    jd_skills = ['java', 'spring boot', 'sql', 'aws', 'git']
    jd_min_exp = 5.0
    jd_edu_level = 2 # Bachelor
    
    # Preprocess
    c_combined = (profile['name'] + " " + " ".join(profile['skills']) + " " + profile['text'])
    j_combined = (jd_title + " " + jd_text)
    
    c_tfidf = tfidf.transform([c_combined])
    j_tfidf = tfidf.transform([j_combined])
    sim = cosine_similarity(c_tfidf, j_tfidf)[0][0]
    
    overlap = len(set(profile['skills']).intersection(set(jd_skills)))
    ratio = overlap / len(jd_skills)
    
    exp_diff = profile['experience_years'] - jd_min_exp
    edu_diff = profile['education_level'] - jd_edu_level
    
    features = np.array([[sim, overlap, ratio, exp_diff, edu_diff]])
    predicted_score = model.predict(features)[0]
    score_pct = int(np.clip(predicted_score * 100, 0, 100))
    
    print(f"Predicted Match Score: {score_pct}%")
    print(f" - TF-IDF Sim: {sim:.4f}")
    print(f" - Skills overlap: {overlap}/5")
    
    assert 0 <= score_pct <= 100, "Predicted score is out of range!"
    
    # 5. Check local Ollama status
    print("\nChecking local Ollama connection...")
    try:
        res = requests.get("http://localhost:11434/", timeout=1)
        if res.status_code == 200:
            print(" - Local Ollama API: ONLINE (Port 11434)")
            # print models
            m_res = requests.get("http://localhost:11434/api/tags", timeout=1)
            models = [m['name'] for m in m_res.json().get('models', [])]
            print(f" - Available Models: {models}")
        else:
            print(f" - Local Ollama API: OFFLINE (Returned status: {res.status_code})")
    except Exception as e:
        print(f" - Local Ollama API: OFFLINE (Could not connect: {e})")
        print("   (Note: Ollama is optional for parsing but required for the AI chat feature. You can still run the app).")
    
    print("\n[PASSED] All automated integration tests PASSED successfully!")

if __name__ == '__main__':
    run_tests()
