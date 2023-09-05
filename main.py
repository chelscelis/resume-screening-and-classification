import pandas as pd
import streamlit as st 

from utils import *

st.write("""
# Resume Screening & Classification
""")
st.caption("""
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
    """)
    st.divider()
    st.write("""
    ## Input Guide 
    #### For the Job Description: 
    Ensure the job description is saved in a text (.txt) file. 
    Clearly outline the responsibilities, qualifications, and skills associated with the position.

    #### For the Resumes: 
    Resumes must be compiled in an excel (.xlsx) file. 
    The organization of columns is up to you but ensure that the "Resume" column is present.
    The values under this column should include all the relevant details of the resume.
    You may refer to the sample format below.
    """)
    with st.expander('View sample format'):
        st.write('# Hello world')
    st.write("""
    #### Demo Files: 
    We have included input files for demo testing purposes. 
    Fifty-five resumes (five resumes per category) were collected from LinkedIn with personally identifiable information (PII) removed.
    Then, eleven job descriptions were collected from Indeed and Glassdoor.
    You can download the following files to experience the capabilities of the web app:

    - **Access Job Description files [here]()**
    - **Access Resume files [here]()**
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
        ##### Upload Resumes:
        - Navigate to the "Classify" tab.
        - Click the "Upload Resumes" button.
        - Select the Excel file (.xlsx) containing the resumes you want to classify. Ensure that your Excel file has the "Resume" column with the resume text and any necessary columns for filtering or additional information.
        ######
        """)
        st.write("""
        ##### Start Processing:
        - Click the "Start Processing" button.
        - The app will analyze the resumes and categorize them into job industry categories.
        ######
        """)
        st.write("""
        ##### View Classification Results:
        - A bar chart will appear, showing the number of resumes per category, helping you visualize the distribution.
        - Below the bar chart, there is a dataframe that displays the list of resumes along with their assigned categories.
        ######
        """)
        st.write("""
        ##### Add Filters:
        - You can apply filters to the dataframe to narrow down your results.
        ######
        """)
        st.write("""
        ##### Donwload Results:
        - Once you've applied filters or are satisfied with the results, you can download the current dataframe as an Excel file by clicking the "Save Current Output as XLSX" button.
        ####
        """)
    st.write("""
    #### Rank Tab:
    The web app will rank the resumes based on their similarity to the job description. 
    Keep in mind, the ranking placement does not imply that one applicant is absolutely better than the other.
    Interviews and other tests are still preferred before making final decisions.

    """)
    with st.expander('Ranking Steps'):
        st.write("""
        ##### Upload Job Description and Resumes:
        - Navigate to the "Rank" tab.
        - Upload the job description as a text file. This file should contain the description of the job you want to compare resumes against.
        - Upload the Excel file that contains the resumes you want to rank.
        ######
        """)
        st.write("""
        ##### Start Processing:
        - Click the "Start Processing" button.
        - The app will analyze the job description and rank the resumes based on their similarity to the job description.
        ######
        """)
        st.write("""
        ##### View Classification Results:
        - The output will display the contents of the job description.
        - Below the job description, there is a dataframe that lists the ranked resumes along with their similarity scores.
        ######
        """)
        st.write("""
        ##### Add Filters:
        - You can apply filters to the dataframe to narrow down your results.
        ######
        """)
        st.write("""
        ##### Donwload Results:
        - Once you've applied filters or are satisfied with the results, you can download the current dataframe as an Excel file by clicking the "Save Current Output as XLSX" button.
        ####
        """)
    # TODO: add picture per step
    # TODO: add file links

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

