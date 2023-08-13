import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('stopwords')

# Sample job description
job_description = "We are looking for a software engineer with strong programming skills."

# Sample resumes
resumes = [
    "Experienced software engineer with a background in programming and web development.",
    "Passionate developer with expertise in multiple programming languages and software design.",
    "Software engineering graduate with a focus on algorithms and problem-solving.",
    "Web developer specializing in frontend and backend programming.",
]

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalnum() and word not in stop_words]
    return " ".join(words)

def rank_resumes(job_desc, resume_list):
    preprocessed_desc = preprocess_text(job_desc)
    
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([preprocessed_desc] + resume_list)
    
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    
    ranked_resumes = sorted(
        zip(resume_list, cosine_similarities),
        key=lambda x: x[1],
        reverse=True
    )
    
    return ranked_resumes

ranked_resumes = rank_resumes(job_description, resumes)

print("Ranking of Resumes based on Similarity to Job Description:")
for idx, (resume, similarity) in enumerate(ranked_resumes, start=1):
    print(f"{idx}. Resume: {resume}")
    print(f"   Similarity: {similarity:.4f}")
    print("=" * 40)

