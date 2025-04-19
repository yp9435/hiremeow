import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from wordcloud import WordCloud
import io

# Import NLP utilities
from nlp_utils import (
    preprocess_text,
    extract_job_entities,
    extract_job_requirements,
    extract_job_sections,
    generate_technical_questions,
    generate_behavioral_questions,
    create_wordcloud,
    create_entity_visualization
)

# Set page config
st.set_page_config(
    page_title="AI-Powered Interview Question Generator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4B0082;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #6A5ACD;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.2rem;
        color: #483D8B;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .question-card {
        background-color: #F8F8FF;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        border-left: 4px solid #4B0082;
    }
    .skill-tag {
        background-color: #E6E6FA;
        color: #4B0082;
        padding: 0.3rem 0.6rem;
        border-radius: 1rem;
        margin-right: 0.3rem;
        margin-bottom: 0.3rem;
        display: inline-block;
        font-size: 0.8rem;
    }
    .nlp-technique {
        background-color: #E6E6FA;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin-bottom: 0.3rem;
        font-size: 0.9rem;
    }
    .entity-card {
        background-color: #F0F0FF;
        padding: 0.8rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        border-left: 4px solid #6A5ACD;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        font-size: 0.8rem;
        color: #6A5ACD;
    }
    .relevance-badge {
        float: right;
        background-color: #4B0082;
        color: white;
        padding: 0.2rem 0.4rem;
        border-radius: 0.3rem;
        font-size: 0.7rem;
    }
    .requirement-card {
        background-color: #F8F8FF;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #6A5ACD;
    }
    .requirement-title {
        font-weight: bold;
        font-size: 1.1rem;
        color: #4B0082;
    }
    .entity-label {
        font-weight: bold;
        color: #6A5ACD;
    }
    .entity-value {
        background-color: #E6E6FA;
        padding: 0.2rem 0.4rem;
        border-radius: 0.3rem;
        margin-right: 0.3rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Load question bank
@st.cache_data
def load_question_bank():
    try:
        return pd.read_csv('data/question_bank.csv')
    except:
        st.error("Question bank not found. Please make sure 'data/question_bank.csv' exists.")
        return pd.DataFrame(columns=['Dataset ID', 'Question', 'Category', 'Sub-Category', 'Difficulty', 'Source', 'Company'])

# Main application
def main():
    # Header
    st.markdown("<h1 class='main-header'>AI-Powered Interview Question Generator</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Upload a job description and get customized interview questions</p>", unsafe_allow_html=True)
    
    # Sidebar with NLP techniques explanation
    with st.sidebar:
        st.header("NLP Techniques Used")
        techniques = [
            "Named Entity Recognition (NER)", 
            "Part-of-Speech Tagging",
            "Tokenization", 
            "Lemmatization", 
            "TF-IDF Vectorization",
            "Cosine Similarity",
            "Keyword Extraction",
            "Text Classification",
            "N-grams Analysis",
            "Entity Linking"
        ]
        
        for technique in techniques:
            st.markdown(f"<div class='nlp-technique'>{technique}</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("About")
        st.write("""
        This application uses advanced NLP techniques to analyze job descriptions and generate 
        personalized interview questions for candidates based on job requirements.
        
        Upload a job description to get started!
        """)
    
    # Load question bank
    question_bank = load_question_bank()
    
    # Text area for job description
    st.subheader("Enter Job Description")
    job_description = st.text_area("Paste the job description here", height=200)
    
    # File uploader as an alternative
    uploaded_file = st.file_uploader("Or upload a job description file (TXT)", type=["txt"])
    
    if uploaded_file is not None:
        # Read the file
        job_description = uploaded_file.getvalue().decode("utf-8")
        st.success("Job description file uploaded successfully!")
        
        # Show the job description
        with st.expander("View uploaded job description"):
            st.text(job_description)
    
    # Process job description
    if job_description:
        # Extract job entities and requirements
        job_entities = extract_job_entities(job_description)
        job_requirements = extract_job_requirements(job_description)
        job_sections = extract_job_sections(job_description)
        
        # Create tabs for different sections
        tab1, tab2, tab3, tab4 = st.tabs(["Technical Questions", "Behavioral Questions", "Job Analysis", "NLP Visualizations"])
        
        with tab1:
            st.markdown("<h2 class='section-header'>Technical Questions</h2>", unsafe_allow_html=True)
            if st.button("Generate Technical Questions"):
                with st.spinner("Generating technical questions..."):
                    tech_questions = generate_technical_questions(job_description, question_bank)
                    
                    if tech_questions:
                        for i, q_data in enumerate(tech_questions, 1):
                            # Display question with relevance score
                            relevance = q_data.get("relevance", 0) * 100
                            relevance_badge = f"<span class='relevance-badge'>Relevance: {relevance:.0f}%</span>"
                            
                            question_text = q_data.get("question", "")
                            
                            # Display matching skills if available
                            matching_skills = q_data.get("matching_skills", [])
                            skills_html = ""
                            if matching_skills:
                                skills_html = "<div style='margin-top: 0.5rem;'>"
                                skills_html += "Related skills: "
                                for skill in matching_skills:
                                    skills_html += f"<span class='skill-tag'>{skill}</span>"
                                skills_html += "</div>"
                            
                            st.markdown(
                                f"<div class='question-card'>{relevance_badge}<b>Q{i}:</b> {question_text}{skills_html}</div>", 
                                unsafe_allow_html=True
                            )
                    else:
                        st.info("No technical questions could be generated. Try providing a more detailed job description.")
        
        with tab2:
            st.markdown("<h2 class='section-header'>Behavioral Questions</h2>", unsafe_allow_html=True)
            if st.button("Generate Behavioral Questions"):
                with st.spinner("Generating behavioral questions..."):
                    behavioral_questions = generate_behavioral_questions(job_description, question_bank)
                    
                    if behavioral_questions:
                        for i, q_data in enumerate(behavioral_questions, 1):
                            # Display question with theme and relevance
                            theme = q_data.get("theme", "general")
                            relevance = q_data.get("relevance", 0) * 100
                            relevance_badge = f"<span class='relevance-badge'>Relevance: {relevance:.0f}%</span>"
                            
                            question_text = q_data.get("question", "")
                            
                            # Add theme tag
                            theme_html = f"<div style='margin-top: 0.5rem;'><span class='skill-tag'>{theme}</span></div>"
                            
                            st.markdown(
                                f"<div class='question-card'>{relevance_badge}<b>Q{i}:</b> {question_text}{theme_html}</div>", 
                                unsafe_allow_html=True
                            )
                    else:
                        st.info("No behavioral questions could be generated. Try providing a more detailed job description.")
        
        with tab3:
            st.markdown("<h2 class='section-header'>Job Analysis</h2>", unsafe_allow_html=True)
            
            # Display extracted entities
            st.subheader("Extracted Entities")
            if job_entities:
                # Display entities by category
                for entity_type, entities in job_entities.items():
                    if entities:
                        st.write(f"**{entity_type}:**")
                        entities_html = "<div>"
                        for entity in entities:
                            entities_html += f"<span class='entity-value'>{entity}</span>"
                        entities_html += "</div>"
                        st.markdown(entities_html, unsafe_allow_html=True)
            else:
                st.info("No entities detected.")
            
            # Display extracted requirements
            st.subheader("Job Requirements")
            if job_requirements:
                for req_type, requirements in job_requirements.items():
                    if requirements:
                        # Create requirement card
                        req_html = f"<div class='requirement-card'><span class='requirement-title'>{req_type}</span><br><br>"
                        
                        # Add requirements as bullet points
                        for req in requirements:
                            req_html += f"<div>• {req}</div>"
                        
                        req_html += "</div>"
                        st.markdown(req_html, unsafe_allow_html=True)
            else:
                st.info("No specific requirements detected.")
            
            # Display job sections
            st.subheader("Job Description Sections")
            if job_sections:
                for section_name, content in job_sections.items():
                    with st.expander(f"{section_name.title()}"):
                        st.write(content)
            else:
                st.info("No clear sections detected in the job description.")
        
        with tab4:
            st.markdown("<h2 class='section-header'>NLP Visualizations</h2>", unsafe_allow_html=True)
            
            # Word Cloud
            st.subheader("Word Cloud")
            if job_description:
                wordcloud_fig = create_wordcloud(job_description)
                st.pyplot(wordcloud_fig)
            
            # Entity Visualization
            st.subheader("Named Entity Recognition")
            if job_description:
                entity_fig = create_entity_visualization(job_description)
                st.pyplot(entity_fig)
            
            # N-gram Analysis
            st.subheader("Top N-grams")
            if job_description:
                from nltk import ngrams
                from collections import Counter
                
                # Preprocess text
                processed_text = preprocess_text(job_description)
                
                if processed_text:
                    # Create bigrams and trigrams
                    bigrams_list = list(ngrams(processed_text, 2))
                    trigrams_list = list(ngrams(processed_text, 3))
                    
                    # Count frequencies
                    bigram_counts = Counter(bigrams_list)
                    trigram_counts = Counter(trigrams_list)
                    
                    # Get top n-grams
                    top_bigrams = bigram_counts.most_common(10)
                    top_trigrams = trigram_counts.most_common(10)
                    
                    # Create DataFrames
                    if top_bigrams:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Top Bigrams**")
                            bigram_df = pd.DataFrame(
                                [(f"{bg[0][0]} {bg[0][1]}", bg[1]) for bg in top_bigrams],
                                columns=["Bigram", "Count"]
                            )
                            st.dataframe(bigram_df)
                        
                        with col2:
                            st.write("**Top Trigrams**")
                            if top_trigrams:
                                trigram_df = pd.DataFrame(
                                    [(f"{tg[0][0]} {tg[0][1]} {tg[0][2]}", tg[1]) for tg in top_trigrams],
                                    columns=["Trigram", "Count"]
                                )
                                st.dataframe(trigram_df)
                            else:
                                st.info("No trigrams detected.")
                    else:
                        st.info("No n-grams detected.")
    
    # Footer
    st.markdown("<div class='footer'>AI-Powered Interview Question Generator - Powered by Python, NLTK, spaCy, and scikit-learn</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
