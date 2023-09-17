import numpy as np
from gensim.models import KeyedVectors
import nltk
import pandas as pd
import re
from gensim.models import Word2Vec
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

resumeFile1 = '~/Projects/hau/csstudy/resume-screening-and-classification/demo-set/resumes/2482_edited.xlsx'
resumeDf1 = pd.read_excel(resumeFile1)

resumeFile2 = '~/Projects/hau/csstudy/resume-screening-and-classification/demo-set/resumes/resumes.xlsx'
resumeDf2 = pd.read_excel(resumeFile2)

jdFile = '~/Projects/hau/csstudy/resume-screening-and-classification/demo-set/job-descriptions/glassdoor_jobdescriptions.xlsx'
jdDf = pd.read_excel(jdFile)

def preprocessing2(text):
    text = re.sub('http\S+\s*', ' ', text)
    text = re.sub('RT|cc', ' ', text)
    text = re.sub('#\S+', '', text)
    text = re.sub('@\S+', '  ', text)
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)
    text = re.sub(r'[^\x00-\x7f]',r' ', text)
    text = re.sub('\s+', ' ', text)
    words = text.split()
    words = [word for word in words if word.lower() not in stop_words]
    # words = [stemmer.stem(word.lower()) for word in words if word.lower() not in stop_words]
    tagged_words = pos_tag(words)
    lemmatized_words = []
    for word, pos in tagged_words:
        wordnet_pos = get_wordnet_pos(pos)
        lemmatized_words.append(lemmatizer.lemmatize(word.lower(), wordnet_pos))
    return lemmatized_words
    # return words

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

def loadModel():
    # model_path = '~/Projects/hau/csstudy/resume-screening-and-classification/GoogleNews-vectors-negative300.bin'
    # model = KeyedVectors.load_word2vec_format(model_path, binary=True)

    model_path = '~/Projects/hau/csstudy/resume-screening-and-classification/wiki-news-300d-1M.vec'
    # model_path = '~/Projects/hau/csstudy/resume-screening-and-classification/wiki-news-300d-1M-subword.vec'
    model = KeyedVectors.load_word2vec_format(model_path)
    # model = KeyedVectors.load_word2vec_format(model_path, limit=100000)

    # model_path = '~/Projects/hau/csstudy/resume-screening-and-classification/word2vec.bin'
    # model = KeyedVectors.load_word2vec_format(model_path, binary=True)

    return model

model = loadModel()
resumeDf1['cleanedResume'] = resumeDf1['Resume'].apply(lambda x: preprocessing2(x))
resumeDf2['cleanedResume'] = resumeDf2['Resume'].apply(lambda x: preprocessing2(x))
jdDf['cleanedDetails'] = jdDf['Details'].apply(lambda x: preprocessing2(x))
corpus = resumeDf1['cleanedResume'].tolist() + resumeDf2['cleanedResume'].tolist() + jdDf['cleanedDetails'].tolist()
corpus = [word for resume in corpus for word in resume]
filtered_word_vectors = {word: model.get_vector(word) for word in corpus if word in model.key_to_index}
words, vectors = zip(*filtered_word_vectors.items())
model = Word2Vec(vector_size=len(vectors[0]), window=5, min_count=1, sg=0)  # You can adjust parameters as needed
model.build_vocab([words])
model.wv.vectors = np.array(vectors)

model.wv.save_word2vec_format("custom_word2vec.vec", binary=False)
model.wv.save_word2vec_format("custom_word2vec.bin", binary=True)
