import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Load and preprocess data
job_description = "We are looking for a software engineer with experience in Python and machine learning."
resumes = [
    "Experienced software engineer with a background in Python and machine learning.",
    "Frontend developer skilled in HTML, CSS, and JavaScript.",
    "Data scientist proficient in Python and R, with a focus on machine learning models.",
    "Software engineer with experience in Python and machine learning.",
    "Passionate software engineer experienced in developing Python applications and implementing machine learning algorithms.",
    "Machine learning engineer with a strong Python background and a track record of creating effective ML solutions.",
    "Experienced Python developer with a specialization in machine learning and a proven ability to deliver high-quality software.",
    "Software engineer with a keen interest in machine learning and a demonstrated history of developing Python-based applications.",
    "Results-driven programmer with expertise in Python and machine learning techniques, ready to contribute as a software engineer.",
    "Experienced software developer proficient in Python and well-versed in machine learning principles.",
]
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
    return tokens

# Train Word2Vec model
all_texts = [job_description] + resumes
all_tokens = [preprocess_text(text) for text in all_texts]

model = Word2Vec(sentences=all_tokens, vector_size=100, window=5, min_count=1, workers=4)

# Calculate cosine similarity
job_description_embedding = np.mean([model.wv[word] for word in preprocess_text(job_description)], axis=0)
resume_embeddings = [np.mean([model.wv[word] for word in preprocess_text(resume)], axis=0) for resume in resumes]

cosine_similarities = [cosine_similarity([job_description_embedding], [resume_embedding])[0][0] for resume_embedding in resume_embeddings]

# Rank resumes
ranked_resumes = sorted(zip(resumes, cosine_similarities), key=lambda x: x[1], reverse=True)

# Print ranked resumes
for rank, (resume, similarity) in enumerate(ranked_resumes, start=1):
    print(f"Rank {rank}: Similarity = {similarity:.4f}")
    print(resume)
    print("="*40)

