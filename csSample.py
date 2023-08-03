import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download stopwords and punkt tokenizer if not already present
nltk.download("stopwords")
nltk.download("punkt")

def preprocess_text(text):
    # Convert to lowercase and remove punctuation
    text = re.sub(r'[^\w\s]', '', text.lower())
    # Tokenize the words
    words = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word not in stop_words]
    # Rejoin the words to form processed text
    return " ".join(words)

def calculate_cosine_similarity(resume, job_description):
    # Combine both texts
    documents = [resume, job_description]

    # Preprocess the texts
    processed_documents = [preprocess_text(document) for document in documents]

    # Create a bag-of-words representation using CountVectorizer
    vectorizer = CountVectorizer().fit_transform(processed_documents)
    vectors = vectorizer.toarray()

    # Calculate the cosine similarity between the two vectors
    similarity_score = cosine_similarity([vectors[0]], [vectors[1]])

    return similarity_score[0][0]

def rank_resumes(resumes, job_description):
    ranked_resumes = []
    for resume in resumes:
        similarity_score = calculate_cosine_similarity(resume, job_description)
        ranked_resumes.append((resume, similarity_score))

    # Sort the resumes based on similarity score in descending order
    ranked_resumes.sort(key=lambda x: x[1], reverse=True)

    return ranked_resumes

if __name__ == "__main__":
    # Example usage
    job_description = "We are looking for a software engineer with experience in Python and machine learning."
    resumes = [
        "Experienced software engineer with a background in Python and machine learning.",
        "Frontend developer skilled in HTML, CSS, and JavaScript.",
        "Data scientist proficient in Python and R, with a focus on machine learning models.",
        "Software engineer with experience in python and machine learning.",
    ]

    ranked_resumes = rank_resumes(resumes, job_description)

    print("Job Description:\n", job_description)
    print("\nRanked Resumes:")
    for i, (resume, similarity_score) in enumerate(ranked_resumes, start=1):
        print(f"{i}. Similarity Score: {similarity_score:.4f}\n   {resume}\n")

