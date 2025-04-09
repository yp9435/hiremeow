import streamlit as st
import random
import base64

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
set_background("cute_background.png")

# Custom CSS for background, fonts, and styles
page_bg = """
<style>
/* Background image */
body {
    background-image: url('image.png');
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
    color: #5a4a3f;
    font-family: 'Comic Sans MS', cursive, sans-serif;
}

/* Main header */
#title {
    text-align: center;
    color: #865dff;
    font-size: 48px;
    font-weight: bold;
}

/* Subheader */
#subheader {
    text-align: center;
    color: #a084e8;
    font-size: 18px;
    margin-bottom: 40px;
}

/* File uploader */
.css-1kyxreq {
    background-color: rgba(255, 255, 255, 0.8); /* Light background for visibility */
    border: 2px dashed #a084e8;
    border-radius: 10px;
    padding: 10px;
    color: #5a4a3f;
    font-size: 16px;
}

/* Style the container */
div:has(> label[for="upload"]) {
    color: #000000;
    font-weight: bold;
    font-size: 16px;
    background: rgba(255, 255, 255, 0.7);
    padding: 4px 8px;
    border-radius: 6px;
    display: inline-block;
    margin-bottom: 8px;
}

/* Style the label text directly */
label[for="upload"] {
    color: #000000 !important;
    font-weight: bold;
}


/* Buttons */
.stButton button {
    background-color: #dab6fc;
    color: #5a4a3f;
    border: none;
    border-radius: 12px;
    padding: 0.75em 1.5em;
    font-size: 16px;
    margin: 0.5em;
    cursor: pointer;
    transition: background-color 0.3s ease;
}

.stButton button:hover {
    background-color: #b784f9;
    color: white;
}

/* Footer */
footer {
    text-align: center;
    color: #865dff;
    font-size: 16px;
    background: rgba(255, 255, 255, 0.7);
    padding: 8px;
    border-radius: 8px;
    margin-top: 30px;
}

/* Customize text area output */
textarea {
    background-color: #fef7f2;
    color: #5a4a3f;
    border-radius: 10px;
    padding: 10px;
    font-size: 16px;
}
</style>

"""

st.markdown(page_bg, unsafe_allow_html=True)

# Title and subtitle
st.markdown('<div id="title">HireMeow  🐱</div>', unsafe_allow_html=True)
st.markdown('<div id="subheader">Your purr-fect interview buddy! Get ready with questions and a dose of cat fun 🐾✨</div>', unsafe_allow_html=True)

# Resume upload
st.file_uploader("Upload your Resume (PDF or DOCX)", type=["pdf", "docx"])

# Question categories
technical_questions = [
    "What is the difference between Python lists and tuples?",
    "Explain the concept of inheritance in OOP.",
    "What are ACID properties in a database?",
    "Describe how a binary search works.",
    "What is a REST API?",
]

behavioural_questions = [
    "Tell me about a time you faced a challenge at work.",
    "How do you handle tight deadlines?",
    "Describe a situation where you worked in a team.",
    "How do you deal with conflict in the workplace?",
    "What motivates you to do your best at work?",
]

surprise_questions = [
    "If you were a cat, how would you solve coding bugs? 🐱",
    "What's your spirit animal and why?",
    "If you could have any superpower during an interview, what would it be?",
    "Convince me to adopt a cat in under 30 seconds! 🐾",
    "Describe your career journey using only cat puns.",
]

# Function to display questions
def display_questions(category):
    if category == "Technical":
        questions = technical_questions
    elif category == "Behavioural":
        questions = behavioural_questions
    else:
        questions = surprise_questions

    st.subheader(f"{category} Questions")
    for q in questions:
        st.write(f"- {q}")

# Buttons for categories
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Technical Questions"):
        display_questions("Technical")

with col2:
    if st.button("Behavioural Qs"):
        display_questions("Behavioural")

with col3:
    if st.button("Surprise Me! 🐾"):
        random_question = random.choice(surprise_questions)
        st.subheader("Surprise Question")
        st.write(f"{random_question}")

# Footer
st.markdown("---", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align: center;color: black; '>Made with ❤️ and a lot of cat cuddles 🐱</div>",
    unsafe_allow_html=True
)