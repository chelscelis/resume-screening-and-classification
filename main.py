import streamlit as st 

st.write("""
# Resume Screening & Classification
Using K-Nearest Neighbors (KNN) algorithm and Cosine Similarity
""")
st.divider()

st.header('Input')
# TODO: finalize job description file type
uploadedJobDescription = st.file_uploader('Upload Job Description')
uploadedResume = st.file_uploader('Upload Resume', type = 'csv')
isButtonDisabled = True

if all([uploadedJobDescription, uploadedResume]):
    isButtonDisabled = False

if st.button('Start Processing', disabled = isButtonDisabled):
    st.divider()
    st.header('Output')
    # TODO: processing logic here
