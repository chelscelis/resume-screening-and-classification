import pandas as pd
import streamlit as st 

from utils import *

st.write("""
# Resume Screening & Classification
Using K-Nearest Neighbors (KNN) algorithm and Cosine Similarity
######
""")

tab1, tab2, tab3 = st.tabs(['Getting Started', 'Classify', 'Rank'])

with tab1:
    st.write("""
    ## Hello, Welcome!  
    In today's competitive job market, the process of manually screening resumes has become a daunting task for recruiters and hiring managers. 
    The sheer volume of applications received for a single job posting can make it extremely time-consuming to identify the most suitable candidates efficiently. 
    This often leads to missed opportunities and the potential loss of top-tier talent.

    The ***Resume Screening & Classification*** website application aims to help alleviate the challenges posed by manual resume screening. The objectives are:
    - To classify the resumes into their most suitable job industry category
    - To compare the resumes to the job description and rank them by similarity

    ## Input Guide 
    ##### For the Job Description: 
    Ensure the job description is saved in a text (.txt) file. 
    Clearly outline the responsibilities, qualifications, and skills associated with the position.

    ##### For the Resumes: 
    Resumes must be compiled in an excel (.xlsx) file. 
    The organization of columns is up to you but ensure that the "Resume" column is present.
    The values under this column should include all the relevant details of the resume.
    You may refer to the sample format below.
    """)
    with st.expander('View sample format'):
        st.write('# Hello world')
    st.write("""
    ##### Demo Files: 
    We have included input files for demo testing purposes. 
    Fifty-five resumes (five resumes per category) were collected from LinkedIn with personally identifiable information (PII) removed.
    Then, eleven job descriptions were collected from Indeed and Glassdoor.
    You can download the following files to experience the capabilities of the web app:

    - **Access Job Description files [here]()**
    - **Access Resume files [here]()**

    ## Demo Walkthrough

    """)
    # TODO: add file links
    # TODO: add web app walkthrough

with tab2:
    st.header('Input')
    uploadedResumeClf = st.file_uploader('Upload Resumes', type = 'xlsx', key = 'upload-resume-clf')

    if uploadedResumeClf is not None:
        isButtonDisabledClf = False
    else:
        st.session_state.processClf = False 
        isButtonDisabledClf = True

    if 'processClf' not in st.session_state:
        st.session_state.processClf = False

    st.button('Start Processing', on_click = clickClassify, disabled = isButtonDisabledClf, key = 'process-clf')

    if st.session_state.processClf:
        st.divider()
        st.header('Output')
        resumeClf = pd.read_excel(uploadedResumeClf)
        resumeClf = resumesClassify(resumeClf)
        with st.expander('View Bar Chart'):
            barChart = createBarChart(resumeClf)
            st.altair_chart(barChart, use_container_width = True)
        currentClf = filterDataframeClf(resumeClf)
        st.dataframe(currentClf, use_container_width = True, hide_index = True)
        xlsxClf = convertDfToXlsx(currentClf)
        st.download_button(label='Save Current Output as XLSX', data = xlsxClf, file_name = 'Resumes_categorized.xlsx')

with tab3:
    st.header('Input')
    uploadedJobDescriptionRnk = st.file_uploader('Upload Job Description', type = 'txt', key = 'upload-jd-rnk')
    uploadedResumeRnk = st.file_uploader('Upload Resumes', type = 'xlsx', key = 'upload-resume-rnk')

    if all([uploadedJobDescriptionRnk, uploadedResumeRnk]):
        isButtonDisabledRnk = False
    else:
        st.session_state.processRank = False
        isButtonDisabledRnk = True

    if 'processRank' not in st.session_state:
        st.session_state.processRank = False

    st.button('Start Processing', on_click = clickRank, disabled = isButtonDisabledRnk, key = 'process-rnk')

    if st.session_state.processRank:
        st.divider()
        st.header('Output')
        jobDescriptionRnk = uploadedJobDescriptionRnk.read().decode('utf-8')
        resumeRnk = pd.read_excel(uploadedResumeRnk)
        resumeRnk = resumesRank(jobDescriptionRnk, resumeRnk)
        with st.expander('View Job Description'):
            st.write(jobDescriptionRnk)
        currentRnk = filterDataframeRnk(resumeRnk)
        st.dataframe(currentRnk, use_container_width = True, hide_index = True)
        xlsxRnk = convertDfToXlsx(currentRnk)
        st.download_button(label='Save Current Output as XLSX', data = xlsxRnk, file_name = 'Resumes_ranked.xlsx')

