import nltk
import string
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [word for word in tokens if word not in string.punctuation]
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

def rank_resumes(job_description, resumes):
    job_description = preprocess_text(job_description)
    preprocessed_resumes = [preprocess_text(resume) for resume in resumes]
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([job_description] + preprocessed_resumes)
    n_components = min(len(resumes), 20)
    lsa = TruncatedSVD(n_components=n_components)
    lsa_matrix = lsa.fit_transform(tfidf_matrix)
    similarity_scores = cosine_similarity(lsa_matrix[0:1], lsa_matrix[1:])
    rankings = {}
    for i, resume in enumerate(resumes):
        rankings[resume] = similarity_scores[0][i]
    ranked_resumes = sorted(rankings.items(), key=lambda x: x[1], reverse=True)
    return ranked_resumes

if __name__ == "__main__":
    job_description = """
Responsibility:
- Manage the Maintenance Slots to ensure optimization of the required aircraft maintenance and defect rectification objectives.
- Prioritize and Co-ordinate Forward Maintenance Requirements.
- Ensure availability of Aircraft Technical Documentation, (Aircraft Maintenance Manual (AMM) Illustrated Parts Catalogue (IPC) Structure Repair Manual (SRM).
- Ensure Availability of Material & Tooling together with Hangar Availability.
- Ensure Sufficient Competent Manpower.
- Evaluation & Preparation of Work Scopes (Work Package).
- Managing the Maintenance Activities Timeline to ensure Optimisation of Delivery.
- Creating And Controlling Work Packages.
- Managing Information related to Service Bulletins (SB’s) / Service Letters (SL’s).
- Ensuring appropriate communication throughout the delivery of the Maintenance Activities.
- Checking Completed Work Cards for completeness and following procedures to support the closeout and arrange for the issue.
- Post Activity Evaluation of Completed and Closed Maintenance Work packages to Review Opportunities for Improvement & Optimisation.

QUALIFICATIONS
- Bachelor's Degree in Aeronautical Engineer/ Aircraft Maintenance Technology
- 3-4 Yrs experience as Production Planner (Aviation)/ Aircraft Mechanic
- Strong organizational and problem-solving skills
- Excellent communication abilities
    """

    resumesFile = '~/Projects/hau/csstudy/resume-screening-and-classification/demo-set/resumes/resumes.xlsx'
    resumesDf = pd.read_excel(resumesFile)
    resumes = resumesDf['Resume'].values

    ranked_resumes = rank_resumes(job_description, resumes)
    
    print("Ranking of resumes:")
    for i, (resume, score) in enumerate(ranked_resumes, start=1):
        print(f"{i}. Resume: ")
        print(f"   Similarity Score: {score:.4f}\n")

