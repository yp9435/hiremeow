import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from wordcloud import WordCloud
import io
import base64


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
st.set_page_config(page_title="HireMeow - Interview Buddy", layout="wide")

# Function to encode image
def set_background(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Set your background
set_background("catbg.jpg")

# Custom CSS for styling
st.markdown("""
<style>
    /* General body styling */
    body {
        background-color: #F8F0FF; /* Light purple background */
        color: #4B0082; /* Dark purple text */
        font-family: 'Arial', sans-serif; /* Clean and modern font */
    }

    /* Main header */
    #title {
        text-align: center;
        color: #6A0DAD; /* Vibrant purple */
        font-size: 48px;
        font-weight: bold;
        margin-bottom: 20px;
    }

    /* Subheader */
    #subheader {
        text-align: center;
        color: #8A2BE2; /* Medium purple */
        font-size: 18px;
        margin-bottom: 40px;
    }

    /* Section headers */
    .section-header {
        font-size: 1.8rem;
        color: #6A0DAD; /* Vibrant purple */
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        font-weight: bold;
        border-bottom: 2px solid #D8BFD8; /* Light purple border */
        padding-bottom: 5px;
    }

    /* Question cards */
    .question-card {
        background-color: #E6E6FA; /* Lavender background */
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 6px solid #6A0DAD; /* Vibrant purple border */
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1); /* Subtle shadow */
    }

    /* Skill tags */
    .skill-tag {
        background-color: #D8BFD8; /* Light purple */
        color: #4B0082; /* Dark purple */
        padding: 0.3rem 0.6rem;
        border-radius: 15px;
        margin-right: 0.3rem;
        margin-bottom: 0.3rem;
        display: inline-block;
        font-size: 0.9rem;
        font-weight: bold;
    }

    /* Entity cards */
    .entity-card {
        background-color: #E6E6FA; /* Lavender background */
        padding: 0.8rem;
        border-radius: 10px;
        margin-bottom: 0.5rem;
        border-left: 6px solid #6A0DAD; /* Vibrant purple border */
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1); /* Subtle shadow */
    }

    /* Footer */
    .footer {
        text-align: center;
        margin-top: 2rem;
        font-size: 0.8rem;
        color: #6A0DAD; /* Vibrant purple */
    }

    /* Relevance badge */
    .relevance-badge {
        float: right;
        background-color: #6A0DAD; /* Vibrant purple */
        color: white;
        padding: 0.2rem 0.4rem;
        border-radius: 5px;
        font-size: 0.8rem;
        font-weight: bold;
    }

    /* Requirement cards */
    .requirement-card {
        background-color: #E6E6FA; /* Lavender background */
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 6px solid #6A0DAD; /* Vibrant purple border */
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1); /* Subtle shadow */
    }

    /* Requirement title */
    .requirement-title {
        font-weight: bold;
        font-size: 1.2rem;
        color: #6A0DAD; /* Vibrant purple */
    }

    /* Entity labels */
    .entity-label {
        font-weight: bold;
        color: #6A0DAD; /* Vibrant purple */
    }

    /* Entity values */
    .entity-value {
        background-color: #D8BFD8; /* Light purple */
        padding: 0.2rem 0.4rem;
        border-radius: 5px;
        margin-right: 0.3rem;
        display: inline-block;
        font-weight: bold;
        color: #4B0082; /* Dark purple */
    }
</style>
""", unsafe_allow_html=True)

# Load question bank
@st.cache_data
def load_question_bank():
    try:
        file_path = 'question_bank.csv' 
        st.write(f"Looking for file at: {file_path}")
        if not os.path.exists(file_path):
            st.error(f"File not found at: {file_path}")
        return pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Error loading question bank: {e}")
        return pd.DataFrame(columns=['Dataset ID', 'Question', 'Category', 'Sub-Category', 'Difficulty', 'Source', 'Company'])
    
# Main application
def main():
    # Title and subtitle
    st.markdown('<div id="title">HireMeow  🐱</div>', unsafe_allow_html=True)
    st.markdown('<div id="subheader">Your purr-fect interview buddy! Get ready with questions and a dose of cat fun 🐾✨</div>', unsafe_allow_html=True)
    
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
                                f"<div class='question-card'><b>Q{i}:</b> {question_text}{skills_html}</div>", 
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
                            question_text = q_data.get("question", "")
                            
                            # Add theme tag
                            theme_html = f"<div style='margin-top: 0.5rem;'><span class='skill-tag'>{theme}</span></div>"
                            
                            st.markdown(
                                f"<div class='question-card'><b>Q{i}:</b> {question_text}{theme_html}</div>", 
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
    st.markdown("---", unsafe_allow_html=True)
    st.markdown(
        "<div style='text-align: center;color: black; '>Made with ❤️ and a lot of cat cuddles 🐱</div>",
        unsafe_allow_html=True
    )
if __name__ == "__main__":
    main()
