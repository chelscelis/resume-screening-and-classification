import numpy as np
import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity
import time
import pandas as pd

startTime = time.time()

# Updated job description and resumes
job_description = "We are looking for a software engineer with experience in Python and machine learning."
resumesDF = pd.read_csv('./knn-trial/datasets/kaggle-KNN-11k.csv')
resumes = resumesDF['Resume'].tolist()
# resumes = [
#     "Experienced software engineer with a background in Python and machine learning.",
#     "Python developer skilled in machine learning algorithms and software engineering best practices.",
#     "Software engineer experienced in Python and machine learning, dedicated to building efficient applications.",
#     "Passionate software engineer with expertise in Python and a strong interest in machine learning advancements.",
#     "Detail-oriented developer with a background in Python and a deep understanding of machine learning techniques.",
#     "Full-stack developer with proficiency in Python and a focus on creating machine learning-powered applications.",
#     "Machine learning engineer with a strong Python background and a track record of developing successful ML solutions.",
#     "Python developer experienced in machine learning model deployment and maintaining reliable software systems.",
#     "Software engineer with experience in Python and a demonstrated ability to develop machine learning pipelines.",
#     "Motivated programmer with a passion for Python and machine learning, capable of delivering innovative solutions.",
#     "Experienced software developer proficient in Python and well-versed in machine learning principles.",
#     "Software engineer with a solid background in Python and hands-on experience in machine learning implementations.",
#     "Python developer with expertise in machine learning algorithms and a track record of impactful software engineering.",
#     "Frontend developer skilled in HTML, CSS, and JavaScript, with a strong interest in machine learning integration.",
#     "Innovative software engineer with a focus on Python and machine learning, delivering creative solutions.",
#     "Results-driven software engineer with a strong Python foundation and expertise in machine learning techniques.",
#     "Experienced data scientist proficient in Python and R, with a focus on machine learning models.",
#     "Python developer with hands-on experience in machine learning and a commitment to delivering exceptional code.",
#     "Software engineer experienced in Python, machine learning, and collaborative problem-solving.",
#     "Experienced software engineer with expertise in Python and a strong interest in machine learning applications.",
#     "Backend developer skilled in Python and experienced in integrating machine learning solutions.",
#     "Python developer with a passion for machine learning and a history of collaborating on successful software projects.",
#     "Software engineer with a background in Python and a deep understanding of machine learning algorithms.",
#     "Machine learning enthusiast skilled in Python programming and software engineering, eager to contribute.",
#     "Experienced coder with a background in Python and a keen interest in machine learning advancements.",
#     "Software engineer with a strong Python foundation and a knack for developing cutting-edge machine learning applications.",
#     "Data scientist proficient in Python and machine learning, with a track record of delivering data-driven solutions.",
#     "Python developer with experience in developing scalable applications and implementing machine learning techniques.",
#     "Software engineer with proficiency in Python and a track record of integrating machine learning into real-world systems.",
#     "Python developer experienced in machine learning model training and deployment, ready to tackle challenging projects.",
#     "Experienced software engineer with a background in Python and a passion for machine learning research.",
#     "Python developer with a focus on machine learning and a proven ability to deliver high-performance software.",
#     "Full-stack developer with a strong Python foundation and experience in implementing machine learning features.",
#     "Software engineer experienced in Python development and a passion for applying machine learning to complex challenges.",
#     "Python developer skilled in machine learning model evaluation, optimization, and software engineering.",
#     "Machine learning engineer with a strong Python background and a track record of creating effective ML solutions.",
#     "Experienced software engineer with a specialization in Python and a deep interest in machine learning advancements.",
#     "Python developer with a passion for machine learning and a proven ability to deliver innovative software solutions.",
#     "Detail-oriented software engineer with expertise in Python and a commitment to delivering high-quality code.",
#     "Software engineer experienced in Python, machine learning, and data analysis, capable of delivering insightful solutions.",
#     "Experienced programmer with a focus on Python and machine learning, dedicated to creating efficient software systems.",
#     "Python developer with a machine learning background, capable of designing and implementing sophisticated software.",
#     "Software engineer with experience in Python and a commitment to code excellence, ready to drive innovation.",
#     "Python developer with hands-on experience in machine learning model development and deployment.",
#     "Innovative software engineer specializing in Python and machine learning, bringing fresh ideas to the table.",
#     "Software engineer skilled in Python and machine learning techniques, capable of developing robust applications.",
#     "Python developer with a focus on data-driven applications and a strong understanding of machine learning concepts.",
#     "Experienced software developer proficient in Python and well-versed in machine learning principles.",
#     "Software engineer with a solid background in Python and hands-on experience in machine learning implementations.",
#     "Passionate software engineer experienced in developing Python applications and implementing machine learning algorithms.",
#     "Python developer experienced in machine learning algorithms and software engineering best practices.",
#     "Motivated software engineer with a background in Python, machine learning, and a commitment to code quality.",
#     "Frontend developer skilled in HTML, CSS, and JavaScript, with an interest in machine learning integration.",
#     "Detail-oriented developer with a background in Python and a deep understanding of machine learning techniques.",
#     "Experienced software engineer with a specialization in Python and a proven ability to deliver high-quality software.",
#     "Python developer with expertise in machine learning algorithms and a track record of successful software engineering.",
#     "Full-stack developer with a focus on Python and machine learning, capable of end-to-end application development.",
#     "Software engineer experienced in Python, machine learning, and collaborative problem-solving.",
#     "Innovative software engineer specializing in Python and machine learning, bringing fresh ideas to the table.",
#     "Python developer experienced in machine learning model deployment and maintaining reliable software systems.",
#     "Machine learning engineer with a strong Python background and a track record of creating effective ML solutions.",
#     "Software engineer with a keen interest in machine learning and a demonstrated history of developing Python-based applications.",
#     "Results-driven programmer with expertise in Python and machine learning techniques, ready to contribute as a software engineer.",
#     "Python developer with a machine learning background, capable of designing and implementing sophisticated software.",
#     "Passionate software engineer experienced in developing Python applications and implementing machine learning algorithms.",
#     "Python developer with a focus on data-driven applications and a strong understanding of machine learning concepts.",
#     "Frontend developer skilled in HTML, CSS, and JavaScript, with a strong interest in machine learning integration.",
#     "Detail-oriented developer with a background in Python and a deep understanding of machine learning techniques.",
#     "Experienced software engineer with a specialization in Python and a proven ability to deliver high-quality software.",
#     "Python developer with expertise in machine learning algorithms and a track record of successful software engineering.",
#     "Full-stack developer with a focus on Python and machine learning, capable of end-to-end application development.",
#     "Software engineer experienced in Python, machine learning, and collaborative problem-solving.",
#     "Innovative software engineer specializing in Python and machine learning, bringing fresh ideas to the table.",
#     "Python developer experienced in machine learning model deployment and maintaining reliable software systems.",
#     "Machine learning engineer with a strong Python background and a track record of creating effective ML solutions.",
#     "Software engineer with a keen interest in machine learning and a demonstrated history of developing Python-based applications.",
#     "Results-driven programmer with expertise in Python and machine learning techniques, ready to contribute as a software engineer.",
# ]

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
