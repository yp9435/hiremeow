import re
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from typing import List, Dict, Union, Tuple, Optional, Any
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams

# Check for sklearn availability
sklearn_available = True
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    sklearn_available = False

# Check for spacy availability
spacy_available = True
try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except:
        import sys
        import subprocess
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")
except ImportError:
    spacy_available = False

# Download necessary NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('taggers/averaged_perceptron_tagger')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')
    nltk.download('punkt', force=True)  # <- Force proper download

# Initialize sentiment analysis pipeline
try:
    from transformers import pipeline
    sentiment_analyzer = pipeline("sentiment-analysis")
except ImportError:
    sentiment_analyzer = None

# Common technical skills list
TECHNICAL_SKILLS = [
    "python", "java", "javascript", "typescript", "c++", "c#", "ruby", "php", "swift", 
    "kotlin", "go", "rust", "scala", "perl", "r", "matlab", "bash", "shell", "c",
    "html", "css", "sass", "less", "bootstrap", "tailwind", "jquery", "react", "angular", 
    "vue", "svelte", "next.js", "gatsby", "node.js", "express", "django", "flask", "spring", 
    "asp.net", "laravel", "symfony", "ruby on rails",
    "sql", "mysql", "postgresql", "mongodb", "oracle", "sqlite", "redis", "cassandra", 
    "dynamodb", "firebase", "neo4j", "elasticsearch", "mariadb", "couchdb",
    "aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "git", "github", "gitlab", 
    "bitbucket", "terraform", "ansible", "puppet", "chef", "circleci", "travis ci", 
    "prometheus", "grafana", "nginx", "apache", "vercel",
    "machine learning", "deep learning", "artificial intelligence", "data science", "nlp", 
    "computer vision", "tensorflow", "pytorch", "keras", "scikit-learn", "pandas", "numpy", 
    "matplotlib", "seaborn", "tableau", "power bi", "hadoop", "spark", "big data", "jupyter",
    "android", "ios", "react native", "flutter", "xamarin", "ionic", "swift", "kotlin", 
    "objective-c", "mobile app development", "expo",
    "agile", "scrum", "kanban", "waterfall", "jira", "confluence", "trello", "asana", 
    "project management", "product management", "sdlc", "devops", "ci/cd",
    "figma", "adobe photoshop", "adobe illustrator", "sketch", "ui design", "ux design", 
    "graphic design", "wireframing", "prototyping", "user research", "spline",
    "rest api", "graphql", "soap", "microservices", "serverless", "blockchain", "web3", 
    "iot", "ar/vr", "game development", "unity", "unreal engine"
]

# Soft skills list
SOFT_SKILLS = [
    "communication", "teamwork", "problem solving", "critical thinking", "leadership",
    "time management", "adaptability", "flexibility", "creativity", "work ethic",
    "interpersonal skills", "attention to detail", "organization", "stress management",
    "decision making", "conflict resolution", "emotional intelligence", "collaboration",
    "negotiation", "persuasion", "presentation", "customer service", "mentoring",
    "coaching", "analytical thinking", "strategic thinking", "innovation", "initiative",
    "self-motivation", "reliability", "accountability", "patience", "empathy"
]

# Preprocess text
def preprocess_text(text: str, remove_stopwords: bool = True) -> List[str]:
    """
    Apply NLP preprocessing techniques to the input text:
    - Tokenization using spaCy
    - Normalization (lowercasing, punctuation removal)
    - Stopword Removal (optional)
    - Lemmatization
    
    Args:
        text: Input text to preprocess
        remove_stopwords: Whether to remove stopwords
        
    Returns:
        List of preprocessed tokens
    """
    if not text or not spacy_available:
        return []
    
    # Tokenization and preprocessing using spaCy
    doc = nlp(text.lower())
    
    # Remove punctuation, numbers, and optionally stopwords
    tokens = [
        token.lemma_ for token in doc
        if not token.is_punct and not token.is_digit and (not remove_stopwords or token.text not in stopwords.words('english'))
    ]
    
    return tokens

# Extract entities from job description using NER
def extract_job_entities(text: str) -> Dict[str, List[str]]:
    """
    Extract entities from job description using Named Entity Recognition.
    
    Args:
        text: Job description text
        
    Returns:
        Dictionary of entity types and their values
    """
    if not text or not spacy_available:
        return {}
    
    # Process the text with spaCy
    doc = nlp(text)
    
    # Initialize entity dictionary
    entities = {
        "Skills": [],
        "Job Titles": [],
        "Organizations": [],
        "Tools": [],
        "Education": [],
        "Experience": []
    }
    
    # Extract named entities
    for ent in doc.ents:
        if ent.label_ == "ORG":
            entities["Organizations"].append(ent.text)
        elif ent.label_ == "PERSON" and len(ent.text.split()) > 1:  # Avoid single names
            entities["Job Titles"].append(ent.text)
        
    # Extract skills using pattern matching
    for skill in TECHNICAL_SKILLS:
        # Create pattern to match whole words only
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text.lower()):
            entities["Skills"].append(skill)
    
    # Extract soft skills
    for skill in SOFT_SKILLS:
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text.lower()):
            if skill not in entities["Skills"]:  # Avoid duplicates
                entities["Skills"].append(skill)
    
    # Extract education requirements
    education_patterns = [
        r'\b(?:bachelor|master|phd|doctorate|bs|ms|ba|ma|mba)\s+(?:degree|in|of)\s+(?:\w+\s+){0,3}(?:science|engineering|arts|business|administration|computer|it|information|technology|mathematics|physics|chemistry|biology)\b',
        r'\b(?:bachelor|master|phd|doctorate)\s+degree\b',
        r'\b(?:bs|ms|ba|ma|mba)\b'
    ]
    
    for pattern in education_patterns:
        matches = re.finditer(pattern, text.lower())
        for match in matches:
            education = match.group(0)
            if education not in entities["Education"]:
                entities["Education"].append(education)
    
    # Extract experience requirements
    experience_patterns = [
        r'\b(\d+)[\+\-]?\s+years?\s+(?:of\s+)?experience\b',
        r'\bexperienced\s+in\b',
        r'\bexperience\s+with\b'
    ]
    
    for pattern in experience_patterns:
        matches = re.finditer(pattern, text.lower())
        for match in matches:
            experience = match.group(0)
            if experience not in entities["Experience"]:
                entities["Experience"].append(experience)
    
    # Extract tools and technologies
    tools_patterns = [
        r'\b(?:tools?|platforms?|frameworks?|libraries|technologies)\s*(?:like|such as|including|:)?\s*((?:[a-zA-Z0-9\-\+\#\.]+(?:\s*,\s*|\s+and\s+|\s+or\s+|\s+)?)+)'
    ]
    
    for pattern in tools_patterns:
        matches = re.finditer(pattern, text.lower())
        for match in matches:
            if len(match.groups()) > 0:
                tools_text = match.group(1)
                # Split by common separators
                tools = re.split(r'\s*,\s*|\s+and\s+|\s+or\s+|\s+', tools_text)
                for tool in tools:
                    if tool and tool.strip() and tool.strip() not in entities["Tools"]:
                        entities["Tools"].append(tool.strip())
    
    # Remove empty categories and duplicates
    entities = {k: list(set(v)) for k, v in entities.items() if v}
    
    return entities

# Extract job requirements
def extract_job_requirements(text: str) -> Dict[str, List[str]]:
    """
    Extract job requirements from the job description.
    
    Args:
        text: Job description text
        
    Returns:
        Dictionary of requirement types and their values
    """
    if not text:
        return {}
    
    # Initialize requirements dictionary
    requirements = {
        "Required Skills": [],
        "Preferred Skills": [],
        "Responsibilities": [],
        "Qualifications": []
    }
    
    # Extract required skills
    required_patterns = [
        r'(?:required skills|requirements|you must have|you should have|required|essential)(?:\s*:|\s*\n)(.*?)(?:\n\n|\n[A-Z]|$)',
        r'(?:^|\n)(?:\*|\-|\d+\.)\s*(.*?(?:experience|proficiency|knowledge|understanding|ability to).*?)(?:\n|$)'
    ]
    
    for pattern in required_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
        for match in matches:
            if len(match.groups()) > 0:
                req_text = match.group(1).strip()
                # Split by bullet points or new lines
                reqs = re.split(r'\n+|\*|\-|\d+\.', req_text)
                for req in reqs:
                    req = req.strip()
                    if req and len(req) > 10 and req not in requirements["Required Skills"]:
                        requirements["Required Skills"].append(req)
    
    # Extract preferred skills
    preferred_patterns = [
        r'(?:preferred skills|nice to have|bonus|plus|preferred|desirable)(?:\s*:|\s*\n)(.*?)(?:\n\n|\n[A-Z]|$)',
        r'(?:^|\n)(?:\*|\-|\d+\.)\s*(.*?(?:preferred|plus|bonus|nice to have).*?)(?:\n|$)'
    ]
    
    for pattern in preferred_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
        for match in matches:
            if len(match.groups()) > 0:
                req_text = match.group(1).strip()
                # Split by bullet points or new lines
                reqs = re.split(r'\n+|\*|\-|\d+\.', req_text)
                for req in reqs:
                    req = req.strip()
                    if req and len(req) > 10 and req not in requirements["Preferred Skills"]:
                        requirements["Preferred Skills"].append(req)
    
    # Extract responsibilities
    responsibility_patterns = [
        r'(?:responsibilities|what you\'ll do|job duties|duties|you will|role|day to day)(?:\s*:|\s*\n)(.*?)(?:\n\n|\n[A-Z]|$)',
        r'(?:^|\n)(?:\*|\-|\d+\.)\s*(.*?(?:responsible for|develop|create|design|implement|manage|lead|collaborate|work with).*?)(?:\n|$)'
    ]
    
    for pattern in responsibility_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
        for match in matches:
            if len(match.groups()) > 0:
                resp_text = match.group(1).strip()
                # Split by bullet points or new lines
                resps = re.split(r'\n+|\*|\-|\d+\.', resp_text)
                for resp in resps:
                    resp = resp.strip()
                    if resp and len(resp) > 10 and resp not in requirements["Responsibilities"]:
                        requirements["Responsibilities"].append(resp)
    
    # Extract qualifications
    qualification_patterns = [
        r'(?:qualifications|education|background|experience required)(?:\s*:|\s*\n)(.*?)(?:\n\n|\n[A-Z]|$)',
        r'(?:^|\n)(?:\*|\-|\d+\.)\s*(.*?(?:degree|education|graduate|certification|qualified|years of experience).*?)(?:\n|$)'
    ]
    
    for pattern in qualification_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
        for match in matches:
            if len(match.groups()) > 0:
                qual_text = match.group(1).strip()
                # Split by bullet points or new lines
                quals = re.split(r'\n+|\*|\-|\d+\.', qual_text)
                for qual in quals:
                    qual = qual.strip()
                    if qual and len(qual) > 10 and qual not in requirements["Qualifications"]:
                        requirements["Qualifications"].append(qual)
    
    # Remove empty categories
    requirements = {k: v for k, v in requirements.items() if v}
    
    return requirements

# Extract job description sections
def extract_job_sections(text: str) -> Dict[str, str]:
    """
    Extract different sections from a job description.
    
    Args:
        text: Job description text
        
    Returns:
        Dictionary of section names and content
    """
    if not text:
        return {}
    
    # Common section names
    section_patterns = {
        "job_summary": r'(?:job summary|about the role|about the job|overview|about the position)(?:\s*:|\s*\n)(.*?)(?:\n\n|\n[A-Z]|$)',
        "responsibilities": r'(?:responsibilities|what you\'ll do|job duties|duties|you will|role|day to day)(?:\s*:|\s*\n)(.*?)(?:\n\n|\n[A-Z]|$)',
        "requirements": r'(?:requirements|qualifications|what you need|what we\'re looking for)(?:\s*:|\s*\n)(.*?)(?:\n\n|\n[A-Z]|$)',
        "preferred": r'(?:preferred|nice to have|bonus|plus|desired|desirable)(?:\s*:|\s*\n)(.*?)(?:\n\n|\n[A-Z]|$)',
        "benefits": r'(?:benefits|perks|what we offer|compensation|why work with us)(?:\s*:|\s*\n)(.*?)(?:\n\n|\n[A-Z]|$)',
        "company": r'(?:about us|about the company|who we are|our company)(?:\s*:|\s*\n)(.*?)(?:\n\n|\n[A-Z]|$)',
        "application": r'(?:how to apply|application process|next steps)(?:\s*:|\s*\n)(.*?)(?:\n\n|\n[A-Z]|$)'
    }
    
    # Initialize sections dictionary
    sections = {}
    
    # Extract each section
    for section_name, pattern in section_patterns.items():
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            sections[section_name] = match.group(1).strip()
    
    return sections

# Generate technical questions based on job description
def generate_technical_questions(job_description: str, question_bank: pd.DataFrame, num_questions: int = 5) -> List[Dict[str, Union[str, float, List[str]]]]:
    """
    Generate technical questions based on job description requirements.
    
    Args:
        job_description: Job description text
        question_bank: DataFrame with question bank data
        num_questions: Number of questions to generate
        
    Returns:
        List of question dictionaries with relevance scores
    """
    if not job_description or question_bank.empty:
        return []
    
    # Extract entities and requirements
    job_entities = extract_job_entities(job_description)
    job_requirements = extract_job_requirements(job_description)
    
    # Get skills from entities
    skills = job_entities.get("Skills", [])
    
    # Get required skills from requirements
    required_skills = []
    for req in job_requirements.get("Required Skills", []):
        # Extract skills from requirements text
        for skill in TECHNICAL_SKILLS:
            if skill.lower() in req.lower() and skill not in required_skills:
                required_skills.append(skill)
    
    # Combine all skills
    all_skills = list(set(skills + required_skills))
    
    # If no skills found, return empty list
    if not all_skills:
        return []
    
    # Filter technical questions from the question bank
    technical_questions = question_bank[question_bank['Category'] == 'Technical'].copy()
    
    if technical_questions.empty:
        return []
    
    # Create a mapping of skills to relevant subcategories
    skill_to_subcategory = {
        "python": "Python",
        "javascript": "JavaScript",
        "typescript": "JavaScript",
        "react": "React",
        "node.js": "Node.js",
        "sql": "SQL",
        "c++": "C++",
        "c": "C",
        "java": "Java",
        "machine learning": "Machine Learning",
        "tensorflow": "Machine Learning",
        "docker": "DevOps",
        "kubernetes": "DevOps",
        "git": "DevOps",
        "aws": "Cloud",
        "azure": "Cloud",
        "firebase": "Database"
    }
    
    # Use advanced NLP techniques if sklearn is available
    if sklearn_available:
        # Create TF-IDF vectorizer for more accurate matching
        tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=10000)
        
        # Process each skill and find matching questions
        matching_questions = []
        
        for skill in all_skills:
            # Find subcategory for this skill
            subcategory = None
            for skill_key, category in skill_to_subcategory.items():
                if skill_key in skill.lower() or skill.lower() in skill_key:
                    subcategory = category
                    break
            
            # Filter questions by skill if possible
            if subcategory and 'Sub-Category' in technical_questions.columns:
                # Try exact match first
                skill_questions = technical_questions[
                    technical_questions['Sub-Category'].str.lower().str.contains(subcategory.lower(), na=False)
                ]
                
                # If not enough questions, use all technical questions
                if len(skill_questions) < 2:
                    skill_questions = technical_questions
            else:
                skill_questions = technical_questions
            
            # Create documents for vectorization
            question_docs = skill_questions['Question'].tolist()
            question_contexts = []
            for q in question_docs:
                question_context = f"{skill} {q}"
                question_contexts.append(question_context)
            
            # Create skill context
            skill_context = f"{skill}"
            
            # Add job requirements that mention this skill
            for req_type, reqs in job_requirements.items():
                for req in reqs:
                    if skill.lower() in req.lower():
                        skill_context += f" {req}"
            
            all_docs = [skill_context] + question_contexts
            
            try:
                # Compute TF-IDF matrix
                tfidf_matrix = tfidf_vectorizer.fit_transform(all_docs)
                
                # Compute cosine similarity between skill and questions
                cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
                
                # Match questions to this skill
                for i, similarity in enumerate(cosine_similarities):
                    if similarity > 0.1:  # Set minimum similarity threshold
                        matching_questions.append({
                            "question": question_docs[i],
                            "skill": skill,
                            "relevance": float(similarity),
                            "subcategory": subcategory
                        })
            except Exception as e:
                print(f"Error in question matching: {e}")
                # Fallback - add questions without similarity score
                for q in question_docs[:min(3, len(question_docs))]:
                    matching_questions.append({
                        "question": q,
                        "skill": skill,
                        "relevance": 1.0,
                        "subcategory": subcategory
                    })
        
        # Sort by relevance
        matching_questions.sort(key=lambda x: x['relevance'], reverse=True)
        
        # Group similar questions and select diverse set
        unique_questions = []
        question_texts = set()
        
        for q_data in matching_questions:
            q_text = q_data['question']
            
            # Skip if we already have this question
            if q_text in question_texts:
                continue
                
            # Check for very similar questions (80% similarity)
            skip = False
            for existing_q in question_texts:
                if existing_q and q_text:
                    similarity = 0
                    if sklearn_available:
                        # Use TF-IDF and cosine similarity for better comparison
                        vec = TfidfVectorizer().fit_transform([existing_q, q_text])
                        similarity = cosine_similarity(vec)[0, 1]
                    else:
                        # Simple Jaccard similarity as fallback
                        set1 = set(existing_q.lower().split())
                        set2 = set(q_text.lower().split())
                        if set1 and set2:
                            similarity = len(set1.intersection(set2)) / len(set1.union(set2))
                
                if similarity > 0.8:
                    skip = True
                    break
                    
            if not skip:
                # Add to unique questions
                question_texts.add(q_text)
                
                # Personalize the question
                personalized_q = q_text
                skill_name = q_data['skill']
                
                # Check if the question already mentions the skill
                if skill_name.lower() not in personalized_q.lower():
                    # Try to insert the skill name in a natural way
                    if "this" in personalized_q:
                        personalized_q = personalized_q.replace("this", f"this {skill_name}")
                    elif "the" in personalized_q:
                        personalized_q = personalized_q.replace("the", f"the {skill_name}")
                    elif "in " in personalized_q:
                        personalized_q = personalized_q.replace("in ", f"in {skill_name} ")
                    elif "of " in personalized_q:
                        personalized_q = personalized_q.replace("of ", f"of {skill_name} ")
                    elif "?" in personalized_q:
                        personalized_q = personalized_q.replace("?", f" in {skill_name}?")
                
                # Add job context if available
                job_context = ""
                for req_type, reqs in job_requirements.items():
                    for req in reqs:
                        if skill_name.lower() in req.lower():
                            job_context = f" This is important because the job requires: '{req}'"
                            break
                    if job_context:
                        break
                
                if job_context and len(personalized_q + job_context) < 200:
                    personalized_q += job_context
                
                unique_questions.append({
                    "question": personalized_q,
                    "original_question": q_text,
                    "relevance": q_data['relevance'],
                    "matching_skills": [q_data['skill']]
                })
                
                # Stop if we have enough questions
                if len(unique_questions) >= num_questions:
                    break
        
        # If we don't have enough questions, add some general technical questions
        if len(unique_questions) < num_questions:
            # Get questions that aren't already selected
            remaining_questions = [q for q in technical_questions['Question'].tolist() if q not in question_texts]
            
            # Randomly select additional questions
            import random
            additional_needed = num_questions - len(unique_questions)
            if remaining_questions and additional_needed > 0:
                additional_questions = random.sample(remaining_questions, min(additional_needed, len(remaining_questions)))
                
                for q_text in additional_questions:
                    unique_questions.append({
                        "question": q_text,
                        "original_question": q_text,
                        "relevance": 0.5,  # Lower relevance for general questions
                        "matching_skills": []
                    })
        
        # Return personalized questions
        return unique_questions[:num_questions]
    
    # Simple fallback if sklearn is not available
    else:
        import random
        question_docs = technical_questions['Question'].tolist()
        
        if len(question_docs) <= num_questions:
            return [{"question": q, "relevance": 1.0, "matching_skills": []} for q in question_docs]
        
        # Select random questions
        sampled_questions = random.sample(question_docs, num_questions)
        return [{"question": q, "relevance": 1.0, "matching_skills": []} for q in sampled_questions]

# Generate behavioral questions based on job description
def generate_behavioral_questions(job_description: str, question_bank: pd.DataFrame, num_questions: int = 5) -> List[Dict[str, Union[str, float, List[str]]]]:
    """
    Generate behavioral questions based on job description.
    
    Args:
        job_description: Job description text
        question_bank: DataFrame with question bank data
        num_questions: Number of questions to generate
        
    Returns:
        List of personalized behavioral questions
    """
    if not job_description or question_bank.empty:
        return []
    
    # Extract job requirements
    job_requirements = extract_job_requirements(job_description)
    
    # Extract job entities
    job_entities = extract_job_entities(job_description)
    
    # Filter behavioral questions from the question bank
    behavioral_questions = question_bank[question_bank['Category'] == 'Behavioral'].copy()
    
    if behavioral_questions.empty:
        return []
    
    # Calculate key behavioral themes from job description
    behavioral_themes = set()
    leadership_evidence = False
    teamwork_evidence = False
    problem_solving_evidence = False
    communication_evidence = False
    innovation_evidence = False
    conflict_resolution_evidence = False
    
    # Look for theme evidence in job requirements
    for req_type, reqs in job_requirements.items():
        for req in reqs:
            req_lower = req.lower()
            
            # Check for leadership evidence
            if any(term in req_lower for term in ["lead", "manage", "direct", "supervise", "oversee"]):
                leadership_evidence = True
                behavioral_themes.add("leadership")
                
            # Check for teamwork evidence
            if any(term in req_lower for term in ["team", "collaborate", "cross-functional", "partner"]):
                teamwork_evidence = True
                behavioral_themes.add("teamwork")
                
            # Check for problem-solving evidence
            if any(term in req_lower for term in ["solve", "solution", "resolve", "fix", "debug", "improve"]):
                problem_solving_evidence = True
                behavioral_themes.add("problem-solving")
                
            # Check for communication evidence
            if any(term in req_lower for term in ["present", "communicate", "report", "document", "write"]):
                communication_evidence = True
                behavioral_themes.add("communication")
                
            # Check for innovation evidence
            if any(term in req_lower for term in ["innovat", "creat", "design", "develop", "new"]):
                innovation_evidence = True
                behavioral_themes.add("innovation")
                
            # Check for conflict resolution evidence
            if any(term in req_lower for term in ["conflict", "disagreement", "resolv", "negotiat", "mediat"]):
                conflict_resolution_evidence = True
                behavioral_themes.add("conflict-resolution")
    
    # Check soft skills for additional themes
    soft_skills = job_entities.get("Skills", [])
    for skill in soft_skills:
        skill_lower = skill.lower()
        
        if "leadership" in skill_lower or "management" in skill_lower:
            behavioral_themes.add("leadership")
        elif "team" in skill_lower or "collaboration" in skill_lower:
            behavioral_themes.add("teamwork")
        elif "problem" in skill_lower or "analytical" in skill_lower:
            behavioral_themes.add("problem-solving")
        elif "communication" in skill_lower or "presentation" in skill_lower:
            behavioral_themes.add("communication")
        elif "innovation" in skill_lower or "creative" in skill_lower:
            behavioral_themes.add("innovation")
        elif "conflict" in skill_lower or "negotiation" in skill_lower:
            behavioral_themes.add("conflict-resolution")
        elif "time" in skill_lower or "deadline" in skill_lower:
            behavioral_themes.add("time-management")
        elif "adapt" in skill_lower or "flexible" in skill_lower:
            behavioral_themes.add("adaptability")
    
    # Personalize questions based on evidence in job description
    personalized_questions = []
    
    # Check if we have Sub-Category for better matching
    has_subcategory = 'Sub-Category' in behavioral_questions.columns
    
    # Create job context for vectorization
    job_context = ""
    for req_type, reqs in job_requirements.items():
        job_context += " ".join(reqs) + " "
    
    # Assign behavioral themes to questions if available
    if has_subcategory:
        # Try to find questions for each theme with evidence
        theme_questions = {}
        
        for theme in behavioral_themes:
            # Filter questions related to this theme
            theme_q = behavioral_questions[
                behavioral_questions['Sub-Category'].str.lower().str.contains(theme, na=False)
            ]
            
            if not theme_q.empty:
                theme_questions[theme] = theme_q
        
        # If we have matching theme questions, prioritize those
        if theme_questions:
            for theme, questions in theme_questions.items():
                # Add theme questions until we reach our limit
                for _, row in questions.iterrows():
                    question_text = row['Question']
                    
                    # Personalize the question with evidence from job description
                    personalized_question = personalize_behavioral_question(question_text, job_description, job_requirements, theme)
                    
                    personalized_questions.append({
                        "question": personalized_question,
                        "original_question": question_text,
                        "theme": theme,
                        "relevance": 1.0,
                        "evidence": True
                    })
                    
                    # Check if we have enough questions
                    if len(personalized_questions) >= num_questions:
                        break
                
                if len(personalized_questions) >= num_questions:
                    break
    
    # If we need more questions, use semantic similarity
    if len(personalized_questions) < num_questions and sklearn_available:
        # Get all behavioral questions not already selected
        remaining_questions = []
        selected_texts = {q["original_question"] for q in personalized_questions}
        
        for _, row in behavioral_questions.iterrows():
            if row['Question'] not in selected_texts:
                remaining_questions.append(row['Question'])
        
        if remaining_questions and job_context:
            try:
                # Create TF-IDF vectorizer
                tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
                
                # Vectorize job context and questions
                all_docs = [job_context] + remaining_questions
                tfidf_matrix = tfidf_vectorizer.fit_transform(all_docs)
                
                # Compute cosine similarity
                cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
                
                # Get top questions by similarity
                top_indices = cosine_similarities.argsort()[-(num_questions - len(personalized_questions)):][::-1]
                
                # Add top questions
                for idx in top_indices:
                    question_text = remaining_questions[idx]
                    
                    # Determine the most likely theme
                    likely_theme = "general"
                    max_similarity = 0
                    
                    for theme in ["leadership", "teamwork", "problem-solving", 
                                 "communication", "innovation", "conflict-resolution",
                                 "time-management", "adaptability"]:
                        if theme.lower() in question_text.lower():
                            likely_theme = theme
                            break
                        
                        # Check similarity to theme keywords
                        theme_similarity = cosine_similarity(
                            tfidf_vectorizer.transform([theme]),
                            tfidf_vectorizer.transform([question_text])
                        )[0, 0]
                        
                        if theme_similarity > max_similarity:
                            max_similarity = theme_similarity
                            likely_theme = theme
                    
                    # Personalize the question
                    personalized_question = personalize_behavioral_question(
                        question_text, job_description, job_requirements, likely_theme
                    )
                    
                    personalized_questions.append({
                        "question": personalized_question,
                        "original_question": question_text,
                        "theme": likely_theme,
                        "relevance": float(cosine_similarities[idx]),
                        "evidence": likely_theme in behavioral_themes
                    })
            except Exception as e:
                print(f"Error generating behavioral questions: {e}")
    
    # If we still need more questions, add random ones
    if len(personalized_questions) < num_questions:
        import random
        remaining_questions = []
        selected_texts = {q["original_question"] for q in personalized_questions}
        
        for _, row in behavioral_questions.iterrows():
            if row['Question'] not in selected_texts:
                remaining_questions.append(row['Question'])
        
        # Randomly sample remaining questions
        num_needed = num_questions - len(personalized_questions)
        if remaining_questions:
            sampled_questions = random.sample(
                remaining_questions, 
                min(num_needed, len(remaining_questions))
            )
            
            for question_text in sampled_questions:
                # Apply simple personalization
                personalized_question = personalize_behavioral_question(
                    question_text, job_description, job_requirements, "general"
                )
                
                personalized_questions.append({
                    "question": personalized_question,
                    "original_question": question_text,
                    "theme": "general",
                    "relevance": 0.5,
                    "evidence": False
                })
    
    # Return the top personalized questions
    personalized_questions.sort(key=lambda x: (x["evidence"], x["relevance"]), reverse=True)
    return personalized_questions[:num_questions]

# Helper function to personalize behavioral questions
def personalize_behavioral_question(question: str, job_description: str, job_requirements: Dict[str, List[str]], theme: str) -> str:
    """
    Personalize a behavioral question with context from job description.
    
    Args:
        question: Original question text
        job_description: Full job description text
        job_requirements: Extracted job requirements
        theme: Behavioral question theme
        
    Returns:
        Personalized question text
    """
    # Start with the original question
    personalized_question = question
    
    # Extract job title if available
    job_title_match = re.search(r'(?:job title|position|role)(?:\s*:|\s*\n)\s*([^\n]+)', job_description, re.IGNORECASE)
    job_title = job_title_match.group(1).strip() if job_title_match else "this position"
    
    # Extract company name if available
    company_match = re.search(r'(?:company|organization|employer)(?:\s*:|\s*\n)\s*([^\n]+)', job_description, re.IGNORECASE)
    company_name = company_match.group(1).strip() if company_match else "our company"
    
    # Replace generic terms with personalized ones
    if "role" in personalized_question.lower():
        personalized_question = re.sub(
            r'\b(your role|your position|your job)\b', 
            f"your role as {job_title}", 
            personalized_question, 
            flags=re.IGNORECASE
        )
    
    if "company" in personalized_question.lower():
        personalized_question = re.sub(
            r'\b(your company|your organization|your team)\b', 
            f"your team at {company_name}", 
            personalized_question, 
            flags=re.IGNORECASE
        )
    
    # Add theme-specific personalization
    if theme == "leadership" and "lead" not in personalized_question.lower():
        personalized_question = personalized_question.replace(
            "?", f" in a leadership capacity?"
        )
    
    elif theme == "teamwork" and "team" not in personalized_question.lower():
        personalized_question = personalized_question.replace(
            "?", f" while working in a team?"
        )
    
    elif theme == "problem-solving" and "problem" not in personalized_question.lower():
        personalized_question = personalized_question.replace(
            "?", f" when faced with a challenging problem?"
        )
    
    # Look for specific requirements to reference
    for req_type, reqs in job_requirements.items():
        for req in reqs:
            # Check if this requirement is relevant to the theme
            if theme == "leadership" and any(term in req.lower() for term in ["lead", "manage", "direct"]):
                personalized_question += f" This is relevant because the job requires: '{req}'"
                break
            elif theme == "teamwork" and any(term in req.lower() for term in ["team", "collaborate"]):
                personalized_question += f" This is relevant because the job requires: '{req}'"
                break
            elif theme == "problem-solving" and any(term in req.lower() for term in ["solve", "solution", "improve"]):
                personalized_question += f" This is relevant because the job requires: '{req}'"
                break
    
    # Remove generic examples if we've added personalization
    personalized_question = re.sub(
        r'$e\.g\.,.*?$', 
        "", 
        personalized_question
    )
    
    return personalized_question

# Create word cloud visualization
def create_wordcloud(text: str):
    """
    Create a word cloud visualization from text.
    
    Args:
        text: Input text
        
    Returns:
        Matplotlib figure with word cloud
    """
    if not text:
        # Return an empty figure if no text
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No data available for word cloud", 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes)
        return fig
    
    # Preprocess text
    processed_text = preprocess_text(text)
    
    if not processed_text:
        # Return an empty figure if no tokens
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No data available for word cloud", 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes)
        return fig
    
    # Create word cloud
    try:
        from wordcloud import WordCloud
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white', 
            colormap='viridis',
            max_words=100,
            contour_width=1,
            contour_color='steelblue'
        ).generate(" ".join(processed_text))
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        plt.tight_layout()
    except ImportError:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "WordCloud library not installed. Please install it to use this function.", 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes)
    
    return fig

# Create entity visualization
def create_entity_visualization(text: str):
    """
    Create a visualization of named entities in the text.
    
    Args:
        text: Input text
        
    Returns:
        Matplotlib figure with entity visualization
    """
    if not text or not spacy_available:
        # Return an empty figure if no text or spaCy not available
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No data available for entity visualization", 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes)
        return fig
    
    # Process text with spaCy
    doc = nlp(text)
    
    # Extract entities
    entities = []
    labels = []
    
    for ent in doc.ents:
        entities.append(ent.text)
        labels.append(ent.label_)
    
    if not entities:
        # Return an empty figure if no entities
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No named entities detected in the text", 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes)
        return fig
    
    # Count entity types
    label_counts = Counter(labels)
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort by count
    sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
    x = [label for label, _ in sorted_labels]
    y = [count for _, count in sorted_labels]
    
    # Create bars
    bars = ax.bar(x, y, color='steelblue')
    
    # Add labels and title
    ax.set_xlabel('Entity Type')
    ax.set_ylabel('Count')
    ax.set_title('Named Entities in Job Description')
    
    # Add count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(height)}', ha='center', va='bottom')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    return fig

# Perform sentiment analysis
def perform_sentiment_analysis(texts: List[str]) -> List[Dict[str, Union[str, float]]]:
    """
    Perform sentiment analysis on a list of texts.

    Args:
        texts: List of texts to analyze.

    Returns:
        List of sentiment analysis results. Each result is a dictionary
        containing 'label' (sentiment label) and 'score' (confidence score).
        Returns an empty list if sentiment analysis is not available.
    """
    if sentiment_analyzer is None:
        print("Sentiment analysis pipeline is not available. Please install transformers library.")
        return []

    try:
        # Analyze sentiment for each text
        sentiments = sentiment_analyzer(texts)
        return sentiments
    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        # Return neutral sentiment as fallback
        return [{'label': 'NEUTRAL', 'score': 0.5} for _ in texts]
