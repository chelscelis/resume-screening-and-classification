import re
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def clean_text(text):
    """Removes punctuation and stop words from text."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = text.split()
    stop_words = set()
    stop_words.update(string.punctuation)
    stop_words.update(['a', 'an', 'the'])
    text = [word for word in text if word not in stop_words]
    return ' '.join(text)

def get_tfidf_vectors(job_description, resumes):
    """Creates TFIDF vectors for the job description and resumes."""
    vectorizer = TfidfVectorizer()
    vectorizer.fit([job_description] + resumes)

    job_description_vector = vectorizer.transform([job_description])
    resumes_vectors = vectorizer.transform(resumes)

    return job_description_vector, resumes_vectors

def get_similarity_scores(job_description_vector, resumes_vectors):
    """Calculates the cosine similarity between the job description and resumes."""
    similarity_scores = cosine_similarity(job_description_vector, resumes_vectors)

    return similarity_scores

def rank_resumes(similarity_scores):
    """Ranks the resumes in descending order of similarity to the job description."""
    ranked_resumes = sorted([(i, score) for i, score in enumerate(similarity_scores)], key=lambda x: x[1], reverse=True)

    return ranked_resumes

def main():
    job_description = "Software Engineer with 5+ years of experience in Python and Java. Must have experience in building web applications and mobile apps. Strong problem-solving and analytical skills. Excellent communication and teamwork skills."
    resumes = ["John Doe is a software engineer with 5+ years of experience in Python and Java. He has built web applications and mobile apps for a variety of companies. He is a strong problem-solver and analytical thinker. He is also an excellent communicator and team player.",
               "Jane Doe is a software engineer with 3+ years of experience in Python. She has built web applications for a variety of companies. She is a strong problem-solver and analytical thinker. She is also an excellent communicator.",
               "Peter Smith is a software engineer with 1+ years of experience in Java. He has built mobile apps for a variety of companies. He is a strong problem-solver and analytical thinker. He is also an excellent team player."]

    job_description_vector, resumes_vectors = get_tfidf_vectors(job_description, resumes)
    similarity_scores = get_similarity_scores(job_description_vector, resumes_vectors)
    ranked_resumes = rank_resumes(similarity_scores)

    print("The top 3 resumes are:")
    for i, (index, score) in enumerate(ranked_resumes):
        print(f"{i+1}. {resumes[index]} (score: {np.array2string(score, precision=2)})")

if __name__ == "__main__":
    main()

