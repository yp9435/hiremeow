import spacy
import random

# Load spaCy's English model
nlp = spacy.load("en_core_web_sm")

# Sample behavioral fallback questions
BEHAVIORAL_QUESTIONS = [
    "Tell me about a time you faced a challenge and how you overcame it.",
    "Describe a situation where you worked in a team.",
    "What motivates you to keep learning new technologies?",
    "How do you manage tight deadlines?",
    "Can you share a time when you showed leadership?"
]

# Sample input (can be replaced with resume text)
def load_resume_text():
    return """
    I'm Yeshaswi, a third-year computer science student. I built an AI-powered chatbot during my internship at HeadStarter AI. 
    I also developed a mobile app using Flutter and Firebase. I'm proficient in Python, C++, and TensorFlow. 
    I've worked on data science projects using pandas and NumPy.
    """

# Extract keywords (tech terms, skills, orgs)
def extract_skills_entities(text):
    doc = nlp(text)
    keywords = set()

    for ent in doc.ents:
        if ent.label_ in ["ORG", "PRODUCT", "PERSON", "NORP"]:
            keywords.add(ent.text)

    for token in doc:
        if token.pos_ in ["PROPN", "NOUN"] and not token.is_stop:
            if token.text.lower() not in ["project", "experience"]:
                keywords.add(token.text)

    return list(keywords)

# Generate custom questions
def generate_questions_from_keywords(keywords):
    question_templates = [
        "Can you tell me more about your experience with {}?",
        "What was the most challenging part of working with {}?",
        "How did you use {} in your projects?",
        "Why did you choose to work with {}?",
        "What would you improve about your experience with {}?"
    ]

    questions = []

    for keyword in keywords:
        template = random.choice(question_templates)
        question = template.format(keyword)
        questions.append(question)

    return questions

# Main function
def main():
    resume_text = load_resume_text()
    keywords = extract_skills_entities(resume_text)

    if keywords:
        print("🎯 Custom Questions Based on Resume:")
        for q in generate_questions_from_keywords(keywords):
            print("- " + q)
    else:
        print("💬 Not enough keywords found. Here's a behavioral question instead:")
        print("- " + random.choice(BEHAVIORAL_QUESTIONS))


if __name__ == "__main__":
    main()
