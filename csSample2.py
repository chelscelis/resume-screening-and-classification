from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import numpy as np
import time

startTime = time.time()

# Sample job description and resumes
job_description = "We are looking for a software engineer with experience in Python and machine learning."
resumes = [
    "Experienced software engineer with a background in Python and machine learning.",
    "Frontend developer skilled in HTML, CSS, and JavaScript.",
    "Data scientist proficient in Python and R, with a focus on machine learning models.",
    "Software engineer with experience in python and machine learning.",
]
# Step 1: Text representation using TF-IDF vectorizer
vectorizer = TfidfVectorizer()
job_vector = vectorizer.fit_transform([job_description])
resumes_vectors = vectorizer.transform(resumes)

# Step 2: Cosine similarity calculation
cosine_similarities = cosine_similarity(job_vector, resumes_vectors)

# Step 3 and 4: Finding k nearest neighbors
k = 4  # Number of neighbors to consider
knn = NearestNeighbors(n_neighbors=k, metric='cosine')
knn.fit(resumes_vectors)
distances, indices = knn.kneighbors(job_vector)

# Step 5: Ranking resumes based on similarity
ranked_resumes = [(resumes[i], 1 - distances[0][j]) for j, i in enumerate(indices[0])]

# Step 6: Displaying the ranked resumes
print("Job Description:\n", job_description, "\n")
print("Ranked Resumes:")
for rank, (resume, similarity) in enumerate(ranked_resumes, start=1):
    print(f"Rank {rank}: Similarity Score: {similarity:.4f}")
    print(resume, "\n")

endTime = time.time()
executionTime = endTime - startTime
print(f'Finished in {executionTime:.4f} seconds')
