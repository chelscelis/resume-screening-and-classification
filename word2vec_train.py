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

resumeDf1['cleanedResume'] = resumeDf1['Resume'].apply(lambda x: preprocessing2(x))
resumeDf2['cleanedResume'] = resumeDf2['Resume'].apply(lambda x: preprocessing2(x))
jdDf['cleanedDetails'] = jdDf['Details'].apply(lambda x: preprocessing2(x))

corpus = resumeDf1['cleanedResume'].tolist() + resumeDf2['cleanedResume'].tolist() + jdDf['cleanedDetails'].tolist()

model = Word2Vec(sentences=corpus, vector_size=300, window=5, min_count=1, sg=1)

# model.save('word2vec.model')
model_path = 'word2vec.bin'
model.wv.save_word2vec_format(model_path, binary=True)

vocabulary_size = len(model.wv.key_to_index)
print("Vocabulary Size:", vocabulary_size)

target_word = 'mechanic'
similar_words = model.wv.most_similar(target_word, topn=10)

for word, score in similar_words:
    print(word, score)
