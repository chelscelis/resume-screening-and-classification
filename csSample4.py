import numpy as np
import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity
import time

startTime = time.time()

# Updated job description and resumes
job_description = "We are looking for a software engineer with experience in Python and machine learning."
resumes = [
    "Experienced software engineer with a background in Python and machine learning.",
    "Frontend developer skilled in HTML, CSS, and JavaScript.",
    "Data scientist proficient in Python and R, with a focus on machine learning models.",
    "Software engineer with experience in python and machine learning.",
]

# Download pre-trained Word2Vec word embeddings
word_vectors = api.load("word2vec-google-news-300")

# Step 1: Text representation using word embeddings
job_vector = np.mean([word_vectors[word] for word in job_description.lower().split() if word in word_vectors], axis=0)
resumes_vectors = np.array([np.mean([word_vectors[word] for word in resume.lower().split() if word in word_vectors], axis=0) for resume in resumes])

# Step 2: Cosine similarity calculation
cosine_similarities = cosine_similarity([job_vector], resumes_vectors)

# Step 3: Ranking resumes based on similarity
ranked_resumes = [(resumes[i], similarity) for i, similarity in enumerate(cosine_similarities[0])]

# Step 4: Displaying the ranked resumes with similarity scores
print("Job Description:\n", job_description, "\n")
print("Ranked Resumes:")
for rank, (resume, similarity) in enumerate(sorted(ranked_resumes, key=lambda x: x[1], reverse=True), start=1):
    print(f"Rank {rank}: Similarity Score: {similarity:.4f}")
    print(resume, "\n")

endTime = time.time()
executionTime = endTime - startTime
print(f'Finished in {executionTime:.4f} seconds')
