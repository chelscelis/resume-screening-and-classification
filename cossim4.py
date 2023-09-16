import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
import nltk
from nltk import pos_tag
from nltk.corpus import stopwords
import re
from sklearn.decomposition import TruncatedSVD

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

path_fastText = 'wiki-news-300d-1M.vec'
model = {}
with open(path_fastText, 'r', encoding='utf-8', newline='\n', errors='ignore') as dictionary:
    for line in dictionary:
        tokens = line.rstrip().split(' ')
        model[tokens[0]] = [float(x) for x in tokens[1:]]
        if len(model) == 400000:
            break

job_description = "We are looking for a software engineer with experience in Python, Django, and web development."
resumes = [
    "I am a software developer with expertise in Python and Django.",
    "Experienced web developer with strong skills in Django and Python.",
    "Java developer with no experience in web development.",
    "I am an experienced chef, great with cooking french meals."
]

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

def calculate_doc_embedding(text):
    tokens = text.split()
    tokens = [token for token in tokens if token in model]
    if len(tokens) == 0:
        return None
    doc_embedding = np.mean([model[token] for token in tokens], axis=0)
    return doc_embedding

job_description_embedding = calculate_doc_embedding(preprocessing2(job_description))
resume_embeddings = [calculate_doc_embedding(preprocessing2(resume)) for resume in resumes]

valid_resume_indices = [i for i, embedding in enumerate(resume_embeddings) if embedding is not None]
resume_embeddings = [resume_embeddings[i] for i in valid_resume_indices]
resumes = [resumes[i] for i in valid_resume_indices]

similarity_scores = cosine_similarity([job_description_embedding], resume_embeddings).flatten()

ranked_resumes = [(resume, score) for resume, score in zip(resumes, similarity_scores)]

ranked_resumes.sort(key=lambda x: x[1], reverse=True)

for i, (resume, score) in enumerate(ranked_resumes, start=1):
    print(f"Rank {i}: Similarity Score {score:.4f}\n{resume}\n")

