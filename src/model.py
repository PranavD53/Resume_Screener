import os
import re
import ast
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parser import parse_resume

# Ensure directories exist
os.makedirs('models', exist_ok=True)

print("Starting model training and BERT pre-computation pipeline...")

# ----------------- Part 1: Pre-train/Pre-compute BERT Embeddings on Word Resumes -----------------
print("\n[PART 1] Pre-computing BERT Embeddings from local Word Resumes...")
resume_dir = 'Resumes'
if not os.path.exists(resume_dir):
    print(f"Resumes directory '{resume_dir}' not found. Skipping BERT local document pre-computation.")
else:
    # Scan files
    files = [
        os.path.join(resume_dir, f) for f in os.listdir(resume_dir) 
        if f.lower().endswith(('.docx', '.pdf'))
    ]
    print(f"Found {len(files)} local candidate resumes in the folder.")
    
    # Load High-End BERT model
    print("Loading High-End BERT Model: 'sentence-transformers/all-mpnet-base-v2' (approx. 420MB)...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        print("BERT model loaded successfully.")
        
        # Mean Pooling helper to extract sentence embeddings
        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        def get_embedding(text):
            # Cap text to 512 tokens (standard BERT limit)
            encoded_input = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt')
            with torch.no_grad():
                model_output = model(**encoded_input)
            sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
            return sentence_embeddings[0].numpy()

        candidate_embeddings = {}
        skills_vocab = set()
        
        for idx, fpath in enumerate(files):
            print(f"Processing candidate {idx+1}/{len(files)}: {os.path.basename(fpath)}")
            try:
                # Parse resume to extract profile details and text
                profile = parse_resume(fpath)
                
                # Combine details for semantic BERT representation
                combined_text = (
                    f"Name: {profile['name']}\n"
                    f"Education: {profile['education_degree']}\n"
                    f"Experience: {profile['experience_years']} years\n"
                    f"Skills: {', '.join(profile['skills'])}\n"
                    f"Content: {profile['text']}"
                )
                
                # Generate embedding vector
                vector = get_embedding(combined_text)
                candidate_embeddings[profile['filename']] = vector
                
                # Update skills vocabulary
                skills_vocab.update(profile['skills'])
            except Exception as e:
                print(f"Error embedding {fpath}: {e}")
                
        # Save pre-computed BERT embeddings
        with open('models/bert_embeddings.pkl', 'wb') as f:
            pickle.dump(candidate_embeddings, f)
        print(f"Pre-computed and saved {len(candidate_embeddings)} BERT embeddings to 'models/bert_embeddings.pkl'.")
        
    except Exception as ex:
        print(f"Failed to run BERT pre-computation: {ex}")

# ----------------- Part 2: Train Classical Random Forest Model on Dataset -----------------
print("\n[PART 2] Training ML Match Score Predictor on Kaggle Dataset...")
csv_path = 'resume_data.csv'
if not os.path.exists(csv_path):
    print(f"Dataset '{csv_path}' not found. Please place it in the workspace directory.")
else:
    print("Reading dataset...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows.")
    
    # Compile standard skills vocabulary from dataset
    print("Compiling skills vocabulary from dataset...")
    skills_vocab_set = set()
    for s in df['skills'].dropna():
        try:
            skills_list = ast.literal_eval(s)
            if isinstance(skills_list, list):
                skills_vocab_set.update([x.strip().lower() for x in skills_list if isinstance(x, str)])
        except:
            pass
            
    skills_vocab_list = {s for s in skills_vocab_set if len(s) >= 3 or s in ['c', 'r', 'go', 'js', 'db', 'ip', 'ai', 'ml', 'qa']}
    skills_vocab_list = sorted(list(skills_vocab_list))
    print(f"Compiled skills list: {len(skills_vocab_list)} skills.")
    
    # Save standard skills vocabulary (overwriting if we have a larger list)
    with open('models/skills_vocab.pkl', 'wb') as f:
        pickle.dump(skills_vocab_list, f)
        
    # Preprocess text columns
    df['career_objective'] = df['career_objective'].fillna('')
    df['skills'] = df['skills'].fillna('[]')
    df['responsibilities'] = df['responsibilities'].fillna('')
    df['educationaL_requirements'] = df['educationaL_requirements'].fillna('')
    df['experiencere_requirement'] = df['experiencere_requirement'].fillna('')
    df['responsibilities.1'] = df['responsibilities.1'].fillna('')
    df['skills_required'] = df['skills_required'].fillna('')
    df['degree_names'] = df['degree_names'].fillna('[]')
    
    job_col = [c for c in df.columns if 'job_position' in c][0]
    df[job_col] = df[job_col].fillna('')
    
    # Fit TF-IDF Vectorizer
    print("Fitting TF-IDF Vectorizer...")
    df['resume_combined'] = df['career_objective'] + ' ' + df['skills'].apply(lambda x: ' '.join(ast.literal_eval(x) if x.startswith('[') else [])) + ' ' + df['responsibilities']
    df['job_combined'] = df[job_col] + ' ' + df['educationaL_requirements'] + ' ' + df['experiencere_requirement'] + ' ' + df['responsibilities.1'] + ' ' + df['skills_required']
    
    tfidf = TfidfVectorizer(stop_words='english', max_features=3000)
    all_texts = pd.concat([df['resume_combined'], df['job_combined']])
    tfidf.fit(all_texts)
    
    # Transform
    resume_tfidf = tfidf.transform(df['resume_combined'])
    job_tfidf = tfidf.transform(df['job_combined'])
    
    # Compute Cosine similarity
    cos_sims = []
    for i in range(len(df)):
        sim = cosine_similarity(resume_tfidf[i], job_tfidf[i])[0][0]
        cos_sims.append(sim)
    df['tfidf_similarity'] = cos_sims
    
    # Extract structural differences
    skill_overlaps = []
    skill_overlap_ratios = []
    exp_diffs = []
    edu_diffs = []
    
    from src.parser import extract_skills_from_text, parse_required_experience, parse_education_level
    
    for idx, row in df.iterrows():
        try:
            c_skills = [s.strip().lower() for s in ast.literal_eval(row['skills'])]
        except:
            c_skills = []
        
        if row['skills_required'].strip():
            j_skills = [s.strip().lower() for s in row['skills_required'].split('\n') if s.strip()]
        else:
            j_skills = extract_skills_from_text(row['job_combined'], skills_vocab_list)
            
        overlap = len(set(c_skills).intersection(set(j_skills)))
        ratio = overlap / len(j_skills) if len(j_skills) > 0 else 0.0
        
        # Parse experience duration from dates
        try:
            starts = ast.literal_eval(row['start_dates'])
            ends = ast.literal_eval(row['end_dates'])
            total_months = 0
            for s, e in zip(starts, ends):
                # Simple month calculation
                try:
                    s_parsed = datetime.strptime(s.strip(), '%b %Y')
                    e_parsed = datetime.now() if e.strip().lower() in ['till date', 'current'] else datetime.strptime(e.strip(), '%b %Y')
                    months = (e_parsed.year - s_parsed.year) * 12 + (e_parsed.month - s_parsed.month)
                    if months > 0:
                        total_months += months
                except:
                    pass
            c_exp = round(total_months / 12.0, 1)
        except:
            c_exp = 0.0
            
        j_exp = parse_required_experience(row['experiencere_requirement'])
        exp_diffs.append(c_exp - j_exp)
        
        c_edu = parse_education_level(row['degree_names'])
        j_edu = parse_education_level(row['educationaL_requirements'])
        edu_diffs.append(c_edu - j_edu)
        
        skill_overlaps.append(overlap)
        skill_overlap_ratios.append(ratio)
        
    df['skill_overlap'] = skill_overlaps
    df['skill_overlap_ratio'] = skill_overlap_ratios
    df['exp_diff'] = exp_diffs
    df['edu_diff'] = edu_diffs
    
    # Feature matrix
    feature_cols = ['tfidf_similarity', 'skill_overlap', 'skill_overlap_ratio', 'exp_diff', 'edu_diff']
    X = df[feature_cols].values
    y = df['matched_score'].values
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train regressor
    regressor = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    regressor.fit(X_train, y_train)
    
    # Evaluate
    y_pred = regressor.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Regressor trained. MAE: {mae:.4f} | R2 Score: {r2:.4f}")
    
    # Save TF-IDF assets
    with open('models/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf, f)
    with open('models/match_model.pkl', 'wb') as f:
        pickle.dump(regressor, f)
        
    # Save training metrics
    metrics = {
        'mae': mae,
        'r2': r2,
        'feature_importances': dict(zip(feature_cols, regressor.feature_importances_))
    }
    with open('models/metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)
        
print("\nModel pipeline execution finished successfully.")
