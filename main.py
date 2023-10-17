import pandas as pd
import streamlit as st 

from utils import *

backgroundPattern = """
<style>
[data-testid="stAppViewContainer"] {
    background-color: #0E1117;
    opacity: 1;
    background-image: radial-gradient(#282C34 0.75px, #0E1117 0.75px);
    background-size: 15px 15px;
}
</style>
"""

# backgroundPattern = """
# <style>
# [data-testid="stAppViewContainer"] {
#     background-color: #FFFFFF;
#     opacity: 1;
#     background-image: radial-gradient(#D1D1D1 0.75px, #FFFFFF 0.75px);
#     background-size: 15px 15px;
# }
# </style>
# """

st.markdown(backgroundPattern, unsafe_allow_html=True)

st.write("""
# Resume Screening & Classification
""")
st.caption("""
Using K-Nearest Neighbors (KNN) algorithm and Cosine Similarity
######
""")

tab1, tab2, tab3 = st.tabs(['Getting Started', 'Classify', 'Rank'])

with tab1:
    writeGettingStarted()

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
        resumeColumnsClf = [col for col in resumeClf.columns if col == "Resume"]

        if len(resumeColumnsClf) == 1:
            resumeClf = classifyResumes(resumeClf)
            with st.expander('View Bar Chart'):
                barChart = createBarChart(resumeClf)
                st.altair_chart(barChart, use_container_width = True)
            currentClf = filterDataframeClf(resumeClf)
            st.dataframe(currentClf, use_container_width = True, hide_index = True)
            xlsxClf = convertDfToXlsx(currentClf)
            st.download_button(label='Save Current Output as XLSX', data = xlsxClf, file_name = 'Resumes_categorized.xlsx')
        elif len(resumeColumnsClf) > 1:
            st.error("""
            #### Oops! Something went wrong.

            Multiple "Resume" columns found in the uploaded excel file.

            Kindly specify which one to use by removing the duplicates :)
            """)
        else:
            st.error("""
            #### Oops! Something went wrong.

            The "Resume" column is not found in the uploaded excel file.

            Kindly make sure the column is present :)
            """)

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
        resumeColumnsRnk = [col for col in resumeRnk.columns if col == "Resume"]

        if len(resumeColumnsRnk) == 1:
            resumeRnk = rankResumes(jobDescriptionRnk, resumeRnk)
            with st.expander('View Job Description'):
                st.write(jobDescriptionRnk)
            currentRnk = filterDataframeRnk(resumeRnk)
            st.dataframe(currentRnk, use_container_width = True, hide_index = True)
            xlsxRnk = convertDfToXlsx(currentRnk)
            st.download_button(label='Save Current Output as XLSX', data = xlsxRnk, file_name = 'Resumes_ranked.xlsx')
        elif len(resumeColumnsRnk) > 1:
            st.error("""
            #### Oops! Something went wrong.

            Multiple "Resume" columns found in the uploaded excel file.

            Kindly specify which one to use by removing the duplicates :)
            """)
        else:
            st.error("""
            #### Oops! Something went wrong.

            The "Resume" column is not found in the uploaded excel file.

            Kindly make sure the column is present :)
            """)

