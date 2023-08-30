import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
# from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Load and preprocess data
job_description = "We are looking for a software engineer with experience in Python and machine learning."
resumes = [
    "Experienced software engineer with a background in Python and machine learning.",
    "Python developer skilled in machine learning algorithms and software engineering best practices.",
    "Software engineer experienced in Python and machine learning, dedicated to building efficient applications.",
    "Passionate software engineer with expertise in Python and a strong interest in machine learning advancements.",
    "Detail-oriented developer with a background in Python and a deep understanding of machine learning techniques.",
    "Full-stack developer with proficiency in Python and a focus on creating machine learning-powered applications.",
    "Machine learning engineer with a strong Python background and a track record of developing successful ML solutions.",
    "Python developer experienced in machine learning model deployment and maintaining reliable software systems.",
]
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
    return tokens

# Train Word2Vec model
all_texts = [job_description] + resumes
all_tokens = [preprocess_text(text) for text in all_texts]

# model = Word2Vec(sentences=all_tokens, vector_size=300, window=5, min_count=1, workers=4)
# Load Google's pretrained Word2Vec model from the binary file
model_path = '~/Projects/hau/csstudy/resume-screening-and-classification/GoogleNews-vectors-negative300.bin'  # Provide the path to the downloaded binary file
model = KeyedVectors.load_word2vec_format(model_path, binary=True)

# Calculate cosine similarity
# job_description_embedding = np.mean([model.wv[word] for word in preprocess_text(job_description)], axis=0)
# resume_embeddings = [np.mean([model.wv[word] for word in preprocess_text(resume)], axis=0) for resume in resumes]
job_description_embedding = np.mean([model[word] for word in preprocess_text(job_description) if word in model], axis=0)
resume_embeddings = [np.mean([model[word] for word in preprocess_text(resume) if word in model], axis=0) for resume in resumes]

# cosine_similarities = [cosine_similarity([job_description_embedding], [resume_embedding])[0][0] for resume_embedding in resume_embeddings]
cosine_similarities = [
    cosine_similarity([job_description_embedding], [resume_embedding])[0][0] /
    (np.linalg.norm(job_description_embedding) * np.linalg.norm(resume_embedding))
    for resume_embedding in resume_embeddings
]

# Rank resumes
ranked_resumes = sorted(zip(resumes, cosine_similarities), key=lambda x: x[1], reverse=True)

# Print ranked resumes
for rank, (resume, similarity) in enumerate(ranked_resumes, start=1):
    print(f"Rank {rank}: Similarity = {similarity:.4f}")
    print(resume)
    print("="*40)

