import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors
import numpy as np

job_description = "We are looking for a software engineer with experience in Python and machine learning."
resumes = [
    "I am a software developer with expertise in Python and deep learning.",
    "Experienced machine learning engineer proficient in Python and AI.",
    "Web developer skilled in JavaScript and React.",
    "Chef experienced in cooking french and european meals and organizing menus."
]

nltk.download("stopwords")
nltk.download("punkt")
stop_words = set(stopwords.words("english"))

def preprocess(text):
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalnum() and word not in stop_words]
    return " ".join(words)

job_description = preprocess(job_description)
resumes = [preprocess(resume) for resume in resumes]

def loadModel():
    model_path = '~/Projects/hau/csstudy/resume-screening-and-classification/wiki-news-300d-1M.vec'
    model = KeyedVectors.load_word2vec_format(model_path)
    return model

model = loadModel()

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix_job_desc = tfidf_vectorizer.fit_transform([job_description])

feature_names = tfidf_vectorizer.get_feature_names_out()
sorted_indices = np.argsort(tfidf_matrix_job_desc.toarray()[0])[::-1]
top_words = [feature_names[i] for i in sorted_indices[:5]]  # Get top 5 important words

job_description_vector = np.mean([model[word] for word in top_words if word in model], axis=0)
resume_vectors = [np.mean([model[word] for word in word_tokenize(resume.lower()) if word in model], axis=0) for resume in resumes]

cosine_similarities = [cosine_similarity([job_description_vector], [rv])[0][0] for rv in resume_vectors]

ranked_resumes = sorted(enumerate(resumes), key=lambda x: cosine_similarities[x[0]], reverse=True)

print("Ranked Resumes:")
for rank, (index, resume) in enumerate(ranked_resumes, start=1):
    print(f"Rank {rank}: Similarity {cosine_similarities[index]:.4f}")
    print(resume)
    print()

