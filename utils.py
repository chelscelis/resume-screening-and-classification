import altair as alt
# import datetime
import joblib
import nltk
import numpy as np
import pandas as pd
import re
import streamlit as st 
import time

from gensim.corpora import Dictionary
from gensim.models import KeyedVectors, TfidfModel
from gensim.similarities import SoftCosineSimilarity, SparseTermSimilarityMatrix, WordEmbeddingSimilarityIndex
from io import BytesIO
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from pandas.api.types import is_categorical_dtype, is_numeric_dtype
from PIL import Image
from scipy.sparse import csr_matrix, hstack

nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def addZeroFeatures(matrix):
    maxFeatures = 18038
    numDocs, numTerms = matrix.shape
    missingFeatures = maxFeatures - numTerms
    if missingFeatures > 0:
        zeroFeatures = csr_matrix((numDocs, missingFeatures), dtype=np.float64)
        matrix = hstack([matrix, zeroFeatures])
    return matrix

from sklearn import metrics
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
@st.cache_data(max_entries = 1, show_spinner = False)
def classifyResumes(df):
    # WITH PROGRESS BAR
    # for x in range(1,11):
    progressBar = st.progress(0)
    progressBar.progress(0, text = "Preprocessing data ...")
    startTime = time.time()
    df['cleanedResume'] = df.Resume.apply(lambda x: performStemming(x))
    resumeText = df['cleanedResume'].values
    progressBar.progress(20, text = "Extracting features ...")
    vectorizer = loadTfidfVectorizer()
    wordFeatures = vectorizer.transform(resumeText)
    wordFeaturesWithZeros = addZeroFeatures(wordFeatures)
    progressBar.progress(40, text = "Reducing dimensionality ...")
    finalFeatures = dimensionalityReduction(wordFeaturesWithZeros)
    progressBar.progress(60, text = "Predicting categories ...")
    knn = loadKnnModel()
    predictedCategories = knn.predict(finalFeatures)
    progressBar.progress(80, text = "Finishing touches ...")
    le = loadLabelEncoder()
    df['Industry Category'] = le.inverse_transform(predictedCategories)
    df['Industry Category'] = pd.Categorical(df['Industry Category'])
    df.drop(columns = ['cleanedResume'], inplace = True)
    endTime = time.time()
    elapsedSeconds = endTime - startTime
    hours, remainder = divmod(int(elapsedSeconds), 3600)
    minutes, _ = divmod(remainder, 60)
    secondsWithDecimals = '{:.2f}'.format(elapsedSeconds % 60)
    elapsedTimeStr = f'{hours} h : {minutes} m : {secondsWithDecimals} s'
    progressBar.progress(100, text = f'Classification Complete!')
    time.sleep(1)
    progressBar.empty()
    st.info(f'Finished classifying {len(resumeText)} resumes - {elapsedTimeStr}')
    # actual = df['Actual Category'].values
    # predicted = df['Industry Category'].values
    # print('Accuracy of KNeighbors Classifier: {:.2f}'.format(knn.score(actual, predicted)))
    # print(f"\n Classification report (LinkedIn Set 55 Resumes) for classifier %s:\n%s\n" % (knn, metrics.classification_report(actual, predicted)))
    # print(f"\n Classification report (LiveCareer Test Set 216 Resumes) for classifier %s:\n%s\n" % (knn, metrics.classification_report(actual, predicted)))
    # print(f"\n Classification report #{x} (LinkedIn Set 55 Resumes) for classifier %s:\n%s\n" % (knn, metrics.classification_report(actual, predicted)))
    # print(f"\n Classification report #{x} (LiveCareer Test Set 216 Resumes) for classifier %s:\n%s\n" % (knn, metrics.classification_report(actual, predicted)))
    # confusion_matrix = metrics.confusion_matrix(actual, predicted)
    # plt.figure(figsize=(10, 10))
    # sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # plt.title(f'Confusion Matrix (LinkedIn Set 55 Resumes)')
    # plt.title(f'Confusion Matrix (LiveCareer Test Set 216 Resumes)')
    # plt.title(f'Confusion Matrix #{x} (LinkedIn Set 55 Resumes)')
    # plt.title(f'Confusion Matrix #{x} (LiveCareer Test Set 216 Resumes)')
    # plt.show()
    return df 

    # NO LOADING WIDGET
    # startTime = time.time()
    # df['cleanedResume'] = df.Resume.apply(lambda x: performStemming(x))
    # resumeText = df['cleanedResume'].values
    # vectorizer = loadTfidfVectorizer()
    # wordFeatures = vectorizer.transform(resumeText)
    # wordFeaturesWithZeros = addZeroFeatures(wordFeatures)
    # finalFeatures = dimensionalityReduction(wordFeaturesWithZeros)
    # knn = loadKnnModel()
    # predictedCategories = knn.predict(finalFeatures)
    # le = loadLabelEncoder()
    # df['Industry Category'] = le.inverse_transform(predictedCategories)
    # df['Industry Category'] = pd.Categorical(df['Industry Category'])
    # df.drop(columns = ['cleanedResume'], inplace = True)
    # endTime = time.time()
    # elapsedSeconds = endTime - startTime
    # elapsedTime = datetime.timedelta(seconds = elapsedSeconds)
    # hours, remainder = divmod(elapsedTime.seconds, 3600)
    # minutes, seconds = divmod(remainder, 60)
    # elapsedTimeStr = f"{hours} hr {minutes} min {seconds} sec"
    # st.info(f'Finished in {elapsedTimeStr}')
    # return df 

def clickClassify():
    st.session_state.processClf = True

def clickRank():
    st.session_state.processRank = True

def convertDfToXlsx(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine = 'xlsxwriter')
    df.to_excel(writer, index = False, sheet_name = 'Sheet1')
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format1 = workbook.add_format({'num_format': '0.00'}) 
    worksheet.set_column('A:A', None, format1)  
    writer.close()
    processedData = output.getvalue()
    return processedData

def createBarChart(df):
    valueCounts = df['Industry Category'].value_counts().reset_index()
    valueCounts.columns = ['Industry Category', 'Count']
    newDataframe = pd.DataFrame(valueCounts)
    barChart = alt.Chart(newDataframe,
    ).mark_bar(
        color = '#56B6C2',
        size = 13 
    ).encode(
        x = alt.X('Count:Q', axis = alt.Axis(format = 'd'), title = 'Number of Resumes'),
        y = alt.Y('Industry Category:N', title = 'Category'),
        tooltip = ['Industry Category', 'Count']
    ).properties(
        title = 'Number of Resumes per Category',
    )
    return barChart

def dimensionalityReduction(features):
    nca = joblib.load('nca_model.joblib')
    features = nca.transform(features.toarray())
    return features    

def filterDataframeClf(df: pd.DataFrame) -> pd.DataFrame:
    modify = st.toggle("Add filters", key = 'filter-clf-1')
    if not modify:
        return df
    df = df.copy()
    modificationContainer = st.container()
    with modificationContainer:
        toFilterColumns = st.multiselect("Filter table on", df.columns, key = 'filter-clf-2')
        for column in toFilterColumns:
            left, right = st.columns((1, 20))
            left.write("↳")
            widgetKey = f'filter-clf-{toFilterColumns.index(column)}-{column}'
            if is_categorical_dtype(df[column]):
                userCatInput = right.multiselect(
                    f'Values for {column}',
                    df[column].unique(),
                    default = list(df[column].unique()),
                    key = widgetKey 
                )
                df = df[df[column].isin(userCatInput)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                userNumInput = right.slider(
                    f'Values for {column}',
                    min_value = _min,
                    max_value = _max,
                    value = (_min, _max),
                    step = step,
                    key = widgetKey 
                )
                df = df[df[column].between(*userNumInput)]
            else:
                userTextInput = right.text_input(
                    f'Substring or regex in {column}',
                    key = widgetKey 
                )
                if userTextInput:
                    userTextInput = userTextInput.lower()
                    df = df[df[column].astype(str).str.lower().str.contains(userTextInput)]
    return df

def filterDataframeRnk(df: pd.DataFrame) -> pd.DataFrame:
    modify = st.toggle("Add filters", key = 'filter-rnk-1')
    if not modify:
        return df
    df = df.copy()
    modificationContainer = st.container()
    with modificationContainer:
        toFilterColumns = st.multiselect("Filter table on", df.columns, key = 'filter-rnk-2')
        for column in toFilterColumns:
            left, right = st.columns((1, 20))
            left.write("↳")
            widgetKey = f'filter-rnk-{toFilterColumns.index(column)}-{column}'
            if is_categorical_dtype(df[column]):
                userCatInput = right.multiselect(
                    f'Values for {column}',
                    df[column].unique(),
                    default = list(df[column].unique()),
                    key = widgetKey
                )
                df = df[df[column].isin(userCatInput)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                userNumInput = right.slider(
                    f'Values for {column}',
                    min_value = _min,
                    max_value = _max,
                    value = (_min, _max),
                    step = step,
                    key = widgetKey
                )
                df = df[df[column].between(*userNumInput)]
            else:
                userTextInput = right.text_input(
                    f'Substring or regex in {column}',
                    key = widgetKey
                )
                if userTextInput:
                    userTextInput = userTextInput.lower()
                    df = df[df[column].astype(str).str.lower().str.contains(userTextInput)]
    return df

def getWordnetPos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def loadKnnModel():
    knnModelFileName = f'knn_model.joblib'
    return joblib.load(knnModelFileName)

def loadLabelEncoder():
    labelEncoderFileName = f'label_encoder.joblib'
    return joblib.load(labelEncoderFileName)

def loadTfidfVectorizer():
    tfidfVectorizerFileName = f'tfidf_vectorizer.joblib' 
    return joblib.load(tfidfVectorizerFileName)

def performLemmatization(text):
    text = re.sub('http\S+\s*', ' ', text)
    text = re.sub('RT|cc', ' ', text)
    text = re.sub('#\S+', '', text)
    text = re.sub('@\S+', '  ', text)
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)
    text = re.sub(r'[^\x00-\x7f]',r' ', text)
    text = re.sub('\s+', ' ', text)
    words = word_tokenize(text)
    words = [
        lemmatizer.lemmatize(word.lower(), pos = getWordnetPos(pos)) 
        for word, pos in pos_tag(words) if word.lower() not in stop_words
    ]
    # text = ' '.join(words)
    return words
    # return text

def performStemming(text):
    text = re.sub('http\S+\s*', ' ', text)
    text = re.sub('RT|cc', ' ', text)
    text = re.sub('#\S+', '', text)
    text = re.sub('@\S+', '  ', text)
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)
    text = re.sub(r'[^\x00-\x7f]',r' ', text)
    text = re.sub('\s+', ' ', text)
    words = word_tokenize(text)
    words = [stemmer.stem(word.lower()) for word in words if word.lower() not in stop_words]
    text = ' '.join(words)
    return text 

@st.cache_data
def loadModel():
    # model_path = '~/Projects/hau/csstudy/resume-screening-and-classification/GoogleNews-vectors-negative300.bin'
    # model = KeyedVectors.load_word2vec_format(model_path, binary=True)

    # model_path = '~/Projects/hau/csstudy/resume-screening-and-classification/wiki-news-300d-1M.vec'
    model_path = '~/Projects/hau/csstudy/resume-screening-and-classification/wiki-news-300d-1M-subword.vec'
    model = KeyedVectors.load_word2vec_format(model_path)
    # model = KeyedVectors.load_word2vec_format(model_path, limit = 300000)

    # model_path = '~/Projects/hau/csstudy/resume-screening-and-classification/word2vec.bin'
    # model = KeyedVectors.load_word2vec_format(model_path, binary=True)

    # model_path = '~/Projects/hau/csstudy/resume-screening-and-classification/custom_word2vec.bin'
    # model = KeyedVectors.load_word2vec_format(model_path, binary=True)
    return model

model = loadModel()

# from gensim.similarities.index import AnnoyIndexer
from gensim.similarities.annoy import AnnoyIndexer
from sklearn.neighbors import NearestNeighbors
# @st.cache_data(max_entries = 1, show_spinner = False)
# def rankResumes(text, df):
#     # WITH PROGRESS BAR
#     progressBar = st.progress(0)
#     progressBar.progress(0, text = "Preprocessing data ...")
#     startTime = time.time()
#     jobDescriptionText = performLemmatization(text)
#     df['cleanedResume'] = df['Resume'].apply(lambda x: performLemmatization(x))
#     documents = [jobDescriptionText] + df['cleanedResume'].tolist()
#     # documents = df['cleanedResume'].tolist()
#     progressBar.progress(13, text = "Creating a dictionary ...")
#     dictionary = Dictionary(documents)
#     progressBar.progress(25, text = "Creating a TF-IDF model ...")
#     tfidf = TfidfModel(dictionary = dictionary)
#     progressBar.progress(38, text = "Creating a Similarity Index...")
#
#     words = [word for word, count in dictionary.most_common()]
#     wordVectors = model.vectors_for_all(words, allow_inference=False)
#     indexer = AnnoyIndexer(wordVectors, num_trees=300) 
#     similarityIndex = WordEmbeddingSimilarityIndex(wordVectors, kwargs={'indexer': indexer})
#
#     # similarityIndex = WordEmbeddingSimilarityIndex(model)
#
#     progressBar.progress(50, text = "Creating a Similarity Matrix...")
#     similarityMatrix = SparseTermSimilarityMatrix(similarityIndex, dictionary, tfidf)
#     progressBar.progress(63, text = "Setting up job description as the query ...")
#     query = tfidf[dictionary.doc2bow(jobDescriptionText)]
#     progressBar.progress(75, text = "Calculating semantic similarities ...")
#     index = SoftCosineSimilarity(
#         tfidf[[dictionary.doc2bow(resume) for resume in df['cleanedResume']]],
#         similarityMatrix 
#     )
#     similarities = index[query]
#     progressBar.progress(88, text = "Finishing touches ...")
#     df['Similarity Score'] = similarities
#     df['Rank'] = df['Similarity Score'].rank(ascending=False, method='dense').astype(int)
#     df.sort_values(by='Rank', inplace=True)
#     df.drop(columns = ['cleanedResume'], inplace = True)
#     endTime = time.time()
#     elapsedSeconds = endTime - startTime
#     hours, remainder = divmod(int(elapsedSeconds), 3600)
#     minutes, _ = divmod(remainder, 60)
#     secondsWithDecimals = '{:.2f}'.format(elapsedSeconds % 60)
#     elapsedTimeStr = f'{hours} h : {minutes} m : {secondsWithDecimals} s'
#     progressBar.progress(100, text = f'Classification Complete!')
#     time.sleep(1)
#     progressBar.empty()
#     st.info(f'Finished ranking {len(df)} resumes - {elapsedTimeStr}')
#     return df 

    # NO LOADING WIDGET
    # startTime = time.time()
    # jobDescriptionText = performLemmatization(text)
    # df['cleanedResume'] = df['Resume'].apply(lambda x: performLemmatization(x))
    # documents = [jobDescriptionText] + df['cleanedResume'].tolist()
    # dictionary = Dictionary(documents)
    # tfidf = TfidfModel(dictionary = dictionary)
    # similarityIndex = WordEmbeddingSimilarityIndex(model)
    # similarityMatrix = SparseTermSimilarityMatrix(similarityIndex, dictionary, tfidf)
    # query = tfidf[dictionary.doc2bow(jobDescriptionText)]
    # index = SoftCosineSimilarity(
    #     tfidf[[dictionary.doc2bow(resume) for resume in df['cleanedResume']]],
    #     similarityMatrix 
    # )
    # similarities = index[query]
    # df['Similarity Score'] = similarities
    # df.sort_values(by = 'Similarity Score', ascending = False, inplace = True)
    # df.drop(columns = ['cleanedResume'], inplace = True)
    # endTime = time.time()
    # elapsedSeconds = endTime - startTime
    # elapsedTime = datetime.timedelta(seconds = elapsedSeconds)
    # hours, remainder = divmod(elapsedTime.seconds, 3600)
    # minutes, seconds = divmod(remainder, 60)
    # elapsedTimeStr = f"{hours} hr {minutes} min {seconds} sec"
    # st.info(f'Finished in {elapsedTimeStr}')
    # return df 

# TF-IDF + LSA + COSSIM
# from sklearn.decomposition import TruncatedSVD
# import math
# def resumesRank(jobDescriptionRnk, resumeRnk):
#     jobDescriptionRnk = preprocessing(jobDescriptionRnk)
#     resumeRnk['cleanedResume'] = resumeRnk.Resume.apply(lambda x: preprocessing(x))
#     resumes = resumeRnk['cleanedResume'].values
#     # tfidfVectorizer = TfidfVectorizer(sublinear_tf = True, stop_words = 'english')
#     # tfidfVectorizer = TfidfVectorizer(sublinear_tf = True)
#     # tfidfVectorizer = TfidfVectorizer(stop_words = 'english')
#     tfidfVectorizer = TfidfVectorizer()
#     tfidfMatrix = tfidfVectorizer.fit_transform([jobDescriptionRnk] + list(resumes))
#     num_features = len(tfidfVectorizer.get_feature_names_out())
#     st.write(f"Number of TF-IDF Features: {num_features}")
#     nComponents = math.ceil(len(resumes) * 0.55)
#     # nComponents = math.ceil(num_features * 0.01)
#     # nComponents = 5 
#     st.write(nComponents)
#     # nComponents = len(resumes)
#     lsa = TruncatedSVD(n_components=nComponents)
#     lsaMatrix = lsa.fit_transform(tfidfMatrix)
#     similarityScores = cosine_similarity(lsaMatrix[0:1], lsaMatrix[1:])
#     resumeRnk['Similarity Score (%)'] = similarityScores[0] * 100
#     resumeRnk = resumeRnk.sort_values(by='Similarity Score (%)', ascending=False)
#     del resumeRnk['cleanedResume']
#     return resumeRnk

# 1 BY 1 SOFT COSSIM
def rankResumes(jobDescriptionRnk, resumeRnk):
    jobDescriptionText = performLemmatization(jobDescriptionRnk)
    resumeRnk['cleanedResume'] = resumeRnk['Resume'].apply(lambda x: performLemmatization(x))
    similarityscore = []
    for resume in resumeRnk['cleanedResume']:
        documents = [jobDescriptionText, resume] 
        dictionary = Dictionary(documents)
        tfidf = TfidfModel(dictionary=dictionary)
        words = [word for word, count in dictionary.most_common()]
        wordVectors = model.vectors_for_all(words, allow_inference=False)
        indexer = AnnoyIndexer(wordVectors, num_trees=300) 
        similarityIndex = WordEmbeddingSimilarityIndex(wordVectors, kwargs={'indexer': indexer})
        similarityMatrix = SparseTermSimilarityMatrix(similarityIndex, dictionary, tfidf)
        # value = tfidf[dictionary.doc2bow(resume)]
        query = dictionary.doc2bow(jobDescriptionText)
        index = SoftCosineSimilarity(
            tfidf[[dictionary.doc2bow(resume)]], 
            # tfidf[[dictionary.doc2bow(jobDescriptionText)]], 
            # [dictionary.doc2bow(resume) for resume in resumeRnk['cleanedResume']],
            similarityMatrix, 
        )
        similarities = index[query]
        similarityscore.append(similarities)
    resumeRnk['Similarity Score'] = similarityscore 
    resumeRnk['Rank'] = resumeRnk['Similarity Score'].rank(ascending=False, method='dense').astype(int)
    resumeRnk.sort_values(by='Rank', inplace=True)
    resumeRnk.drop(columns = ['cleanedResume'], inplace = True)
    return resumeRnk 

# TF-IDF SCORE + WORD EMBEDDINGS SCORE
# def resumesRank(jobDescriptionRnk, resumeRnk):
#     def get_word_embedding(text):
#         words = text.split()
#         valid_words = [word for word in text.split() if word in model]
#         if valid_words:
#             return np.mean([model[word] for word in valid_words], axis=0)
#         else:
#             return np.zeros(model.vector_size)
#     jobDescriptionRnk = preprocessing2(jobDescriptionRnk)
#     resumeRnk['cleanedResume'] = resumeRnk.Resume.apply(lambda x: preprocessing2(x))
#     tfidfVectorizer = TfidfVectorizer(sublinear_tf = True, stop_words='english')
#     jobTfidf = tfidfVectorizer.fit_transform([jobDescriptionRnk])
#     jobDescriptionEmbedding = get_word_embedding(jobDescriptionRnk)
#     resumeSimilarities = []
#     for resumeContent in resumeRnk['cleanedResume']:
#         resumeEmbedding = get_word_embedding(resumeContent)
#         similarityFastText = cosine_similarity([jobDescriptionEmbedding], [resumeEmbedding])[0][0]
#         similarityTFIDF = cosine_similarity(jobTfidf, tfidfVectorizer.transform([resumeContent]))[0][0]
#         similarity = (0.6 * similarityTFIDF) + (0.4 * similarityFastText)
#         final_similarity = similarity * 100
#         resumeSimilarities.append(final_similarity)
#     resumeRnk['Similarity Score (%)'] = resumeSimilarities
#     resumeRnk = resumeRnk.sort_values(by='Similarity Score (%)', ascending=False)
#     del resumeRnk['cleanedResume']
#     return resumeRnk

# WORD EMBEDDINGS + COSSIM
# def resumesRank(jobDescriptionRnk, resumeRnk):
#     def get_word_embedding(text):
#         words = text.split()
#         valid_words = [word for word in text.split() if word in model]
#         if valid_words:
#             return np.mean([model[word] for word in valid_words], axis=0)
#         else:
#             return np.zeros(model.vector_size)
#     jobDescriptionRnk = preprocessing2(jobDescriptionRnk)
#     jobDescriptionEmbedding = get_word_embedding(jobDescriptionRnk)
#     resumeRnk['cleanedResume'] = resumeRnk.Resume.apply(lambda x: preprocessing2(x))
#     resumeSimilarities = []
#     for resumeContent in resumeRnk['cleanedResume']:
#         resumeEmbedding = get_word_embedding(resumeContent)
#         similarity = cosine_similarity([jobDescriptionEmbedding], [resumeEmbedding])[0][0]
#         percentageSimilarity = similarity * 100
#         resumeSimilarities.append(percentageSimilarity)
#     resumeRnk['Similarity Score (%)'] = resumeSimilarities
#     resumeRnk = resumeRnk.sort_values(by='Similarity Score (%)', ascending=False)
#     del resumeRnk['cleanedResume']
#     return resumeRnk

# TF-IDF + COSSIM
# def resumesRank(jobDescriptionRnk, resumeRnk):
#     jobDescriptionRnk = preprocessing2(jobDescriptionRnk)
#     resumeRnk['cleanedResume'] = resumeRnk.Resume.apply(lambda x: preprocessing2(x))
#     tfidfVectorizer = TfidfVectorizer(sublinear_tf = True, stop_words='english')
#     jobTfidf = tfidfVectorizer.fit_transform([jobDescriptionRnk])
#     resumeSimilarities = []
#     for resumeContent in resumeRnk['cleanedResume']:
#         resumeTfidf = tfidfVectorizer.transform([resumeContent])
#         similarity = cosine_similarity(jobTfidf, resumeTfidf)
#         percentageSimilarity = (similarity[0][0] * 100)
#         resumeSimilarities.append(percentageSimilarity)
#     resumeRnk['Similarity Score (%)'] = resumeSimilarities
#     resumeRnk = resumeRnk.sort_values(by='Similarity Score (%)', ascending=False)
#     del resumeRnk['cleanedResume']
#     return resumeRnk

def writeGettingStarted():
    st.write("""
    ## Hello, Welcome!  
    In today's competitive job market, the process of manually screening resumes has become a daunting task for recruiters and hiring managers. 
    The sheer volume of applications received for a single job posting can make it extremely time-consuming to identify the most suitable candidates efficiently. 
    This often leads to missed opportunities and the potential loss of top-tier talent.

    The ***Resume Screening & Classification*** website application aims to help alleviate the challenges posed by manual resume screening. 
    The main objectives are:
    - To classify the resumes into their most suitable job industry category
    - To compare the resumes to the job description and rank them by similarity
    """)
    st.divider()
    st.write("""
    ## Input Guide 
    #### For the Job Description: 
    Ensure the job description is saved in a text (.txt) file. 
    Kindly outline the responsibilities, qualifications, and skills associated with the position.

    #### For the Resumes: 
    Resumes must be compiled in an excel (.xlsx) file. 
    The organization of columns is up to you but ensure that the "Resume" column is present.
    The values under this column should include all the relevant details for each resume.
    """)
    st.divider()
    st.write("""
    ## Demo Walkthrough
    #### Classify Tab:
    The web app will classify the resumes into their most suitable job industry category.
    Currently the Category Scope consists of the following:
    """)
    column1, column2 = st.columns(2)
    with column1:
        st.write("""
        - Aviation
        - Business development
        - Culinary
        - Education
        - Engineering
        - Finance
        """)
    with column2:
        st.write("""
        - Fitness
        - Healthcare
        - HR
        - Information Technology
        - Public relations
        """)
    with st.expander('Classification Steps'):
        st.write("""
        ##### Upload Resumes & Start Processing:
        - Navigate to the "Classify" tab.
        - Click the "Upload Resumes" button.
        - Select the Excel file (.xlsx) containing the resumes you want to classify. Ensure that your Excel file has the "Resume" column with the resume text and any necessary columns for filtering or additional information.
        - Click the "Start Processing" button.
        - The app will analyze the resumes and categorize them into job industry categories.
        ######
        """)
        imgClf1 = Image.open('clf-1.png')
        st.image(imgClf1, use_column_width = True, output_format = "PNG")
        st.write("""
        ##### View Bar Chart:
        - A bar chart will appear, showing the number of resumes per category, helping you visualize the distribution.
        ######
        """)
        imgClf2 = Image.open('clf-2.png')
        st.image(imgClf2, use_column_width = True, output_format = "PNG")
        st.write("""
        ##### Add Filters:
        - You can apply filters to the dataframe to narrow down your results.
        ######
        """)
        imgClf3 = Image.open('clf-3.png')
        st.image(imgClf3, use_column_width = True, output_format = "PNG")
        st.write("""
        ##### Donwload Results:
        - Once you've applied filters or are satisfied with the results, you can download the current dataframe as an Excel file by clicking the "Save Current Output as XLSX" button.
        ####
        """)
        imgClf4 = Image.open('clf-4.png')
        st.image(imgClf4, use_column_width = True, output_format = "PNG")
    st.write("""
    #### Rank Tab:
    The web app will rank the resumes based on their semantic similarity to the job description. 
    The similarity score ranges from -1 to 1.
    A score of 1 is achieved when Document A and Document B are identical.

    ##### **Kindly take note:**

    It's important to note that these scores are not absolute and may change when more resumes are added in the comparison.
    The ranking algorithm dynamically adjusts its results based on the entire set of uploaded resumes.
    We recommend considering the scores as a relative measure rather than an absolute determination.
    """)
    with st.expander('Ranking Steps'):
        st.write("""
        ##### Upload Files & Start Processing:
        - Navigate to the "Rank" tab.
        - Upload the job description as a text file. This file should contain the description of the job you want to compare resumes against.
        - Upload the Excel file that contains the resumes you want to rank.
        - Click the "Start Processing" button.
        - The app will analyze the job description and rank the resumes based on their similarity to the job description.
        ######
        """)
        imgRnk1 = Image.open('rnk-1.png')
        st.image(imgRnk1, use_column_width = True, output_format = "PNG")
        st.write("""
        ##### View Job Description:
        - The output will display the contents of the job description for reference.
        ######
        """)
        imgRnk2 = Image.open('rnk-2.png')
        st.image(imgRnk2, use_column_width = True, output_format = "PNG")
        st.write("""
        ##### Add Filters:
        - You can apply filters to the dataframe to narrow down your results.
        ######
        """)
        imgRnk3 = Image.open('rnk-3.png')
        st.image(imgRnk3, use_column_width = True, output_format = "PNG")
        st.write("""
        ##### Donwload Results:
        - Once you've applied filters or are satisfied with the results, you can download the current dataframe as an Excel file by clicking the "Save Current Output as XLSX" button.
        ####
        """)
        imgRnk4 = Image.open('rnk-4.png')
        st.image(imgRnk4, use_column_width = True, output_format = "PNG")

