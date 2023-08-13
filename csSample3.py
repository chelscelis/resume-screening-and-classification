import numpy as np
import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import time

startTime = time.time()

# Sample job description and resumes
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

# Download pre-trained Word2Vec word embeddings
word_vectors = api.load("word2vec-google-news-300")

# Step 1: Text representation using word embeddings
job_vector = np.mean([word_vectors[word] for word in job_description.split() if word in word_vectors], axis=0)
resumes_vectors = np.array([np.mean([word_vectors[word] for word in resume.split() if word in word_vectors], axis=0) for resume in resumes])

# Step 2: Cosine similarity calculation
cosine_similarities = cosine_similarity([job_vector], resumes_vectors)

# Step 3 and 4: Finding all neighbors
k = 4
knn = NearestNeighbors(n_neighbors=k, metric='cosine')
knn.fit(resumes_vectors)
distances, indices = knn.kneighbors([job_vector])

# Step 5: Ranking resumes based on similarity
ranked_resumes = [(resumes[i], 1 - distance) for distance, i in zip(distances[0], indices[0])]

# Step 6: Displaying all the ranked resumes with similarity scores
print("Job Description:\n", job_description, "\n")
print("Ranked Resumes:")
for rank, (resume, similarity) in enumerate(ranked_resumes, start=1):
    print(f"Rank {rank}: Similarity Score: {similarity:.4f}")
    print(resume, "\n")

endTime = time.time()
executionTime = endTime - startTime
print(f'Finished in {executionTime:.4f} seconds')
