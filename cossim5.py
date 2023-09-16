import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
import nltk
from nltk import pos_tag
from nltk.corpus import stopwords
import re
from gensim.models import KeyedVectors
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


job_description = "We are looking for a software engineer with experience in Python, Django, and web development."
resumes = [
    "I am a software developer with expertise in Python and Django.",
    "Experienced web developer with strong skills in Django and Python.",
    "Java developer with no experience in web development.",
    "I am a head chef experienced with cooking french meals."
]

# Preprocess text and calculate document embeddings
def preprocessing2(text):
    text = re.sub('http\S+\s*', ' ', text)
    text = re.sub('RT|cc', ' ', text)
    text = re.sub('#\S+', '', text)
    text = re.sub('@\S+', '  ', text)
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)
    text = re.sub(r'[^\x00-\x7f]',r' ', text)
    text = re.sub('\s+', ' ', text)
    words = text.split()
    tagged_words = pos_tag(words)
    lemmatized_words = []
    for word, pos in tagged_words:
        if word.lower() not in stop_words:
            wordnet_pos = get_wordnet_pos(pos)
            lemmatized_words.append(lemmatizer.lemmatize(word.lower(), wordnet_pos))
    text = ' '.join(lemmatized_words)
    return text

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return 'a'  # Adjective
    elif tag.startswith('N'):
        return 'n'  # Noun
    elif tag.startswith('R'):
        return 'r'  # Adverb
    elif tag.startswith('V'):
        return 'v'  # Verb
    else:
        return 'n'

job_description = preprocessing2(job_description)
resumes = [preprocessing2(resume) for resume in resumes]

vectorizer = TfidfVectorizer()
job_description_matrix = vectorizer.fit_transform([job_description])
resume_matrix = vectorizer.transform(resumes)

lsa = TruncatedSVD(n_components=8)
job_description_embedding = lsa.fit_transform(job_description_matrix)
resume_embeddings = lsa.transform(resume_matrix)

similarity_scores = cosine_similarity(job_description_embedding, resume_embeddings).flatten()

ranked_resumes = [(resume, score) for resume, score in zip(resumes, similarity_scores)]

ranked_resumes.sort(key=lambda x: x[1], reverse=True)

for i, (resume, score) in enumerate(ranked_resumes, start=1):
    print(f"Rank {i}: Similarity Score {score:.4f}\n{resume}\n")
