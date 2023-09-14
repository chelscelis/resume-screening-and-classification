from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors

def loadModel():
    model_path = '~/Projects/hau/csstudy/resume-screening-and-classification/wiki-news-300d-1M.vec'
    model = KeyedVectors.load_word2vec_format(model_path)
    return model

model = loadModel()
job_description = "This is a sample job description for a data analyst position."
resumes = [
    "I am an experienced data analyst with a strong background in data analysis.",
    "I have a degree in computer science and skills in data analysis and statistics.",
    "I am a software engineer with some experience in data analysis tasks.",
]
documents = [job_description] + resumes
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
job_vector = sum(
    model[word] * tfidf_vectorizer.idf_[tfidf_vectorizer.vocabulary_[word]]
    if word in model and word in tfidf_vectorizer.vocabulary_
    else 0
    for word in job_description.split()
)
resume_vectors = [
    sum(
        model[word] * tfidf_vectorizer.idf_[tfidf_vectorizer.vocabulary_[word]]
        if word in model and word in tfidf_vectorizer.vocabulary_
        else 0
        for word in resume.split()
    )
    for resume in resumes
]
similarities = [
    cosine_similarity(job_vector.reshape(1, -1), resume_vector.reshape(1, -1))[0][0]
    for resume_vector in resume_vectors
]
ranking = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)
for rank, similarity in ranking:
    print(f"Resume {rank + 1}: Similarity = {similarity:.4f}")

