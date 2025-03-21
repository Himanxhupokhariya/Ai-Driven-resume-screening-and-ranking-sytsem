import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time  # For adding loading indicators
import io  # For handling file reading errors


# --- Helper Functions ---
@st.cache_data  # Cache the results for faster loading on subsequent runs
def extract_text_from_pdf(file):
    """Extracts text from a PDF file."""
    try:
        pdf = PdfReader(file)
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None


@st.cache_data
def rank_resumes(job_description, resumes):
    """Ranks resumes based on cosine similarity to the job description."""
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()
    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]
    cosine_similarities = cosine_similarity(
        [job_description_vector], resume_vectors
    ).flatten()
    return cosine_similarities


# --- UI Components ---

# Custom CSS for a more professional look
st.markdown(
    """
    <style>
    .big-font {
        font-size:2.5rem !important;
        font-weight: bold;
    }
    .medium-font {
        font-size:1.5rem !important;
    }
    .stTextInput > label {
        font-size: 1.2rem;
    }
    .stFileUploader > div > div:nth-child(2) > div {
        background-color: #f0f2f6; /* So it stands out */
        border: 1px dashed #a1a1a1;
        padding: 10px;
    }
    .dataframe {
        border: 1px solid #ddd;
    }
    .dataframe th {
        background-color: #f2f2f2;
        text-align: left;
        padding: 8px;
    }
    .dataframe td {
        padding: 8px;
        border-top: 1px solid #ddd;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title and Introduction
st.markdown("<p class='big-font'>AI Resume Screening System</p>", unsafe_allow_html=True)
st.markdown(
    "Welcome!  This tool helps you quickly screen resumes and rank candidates based on their fit for a specific job description.  Simply enter the job description and upload the resumes in PDF format."
)

# Job Description Input
st.markdown("<p class='medium-font'>Job Description</p>", unsafe_allow_html=True)
job_description = st.text_area(
    "Enter the job description:",
    height=200,
    placeholder="Paste the job description here...",
)  # Improved UI

# Resume Upload
st.markdown("<p class='medium-font'>Upload Resumes</p>", unsafe_allow_html=True)
uploaded_files = st.file_uploader(
    "Upload PDF files:", type=["pdf"], accept_multiple_files=True
)

# --- Main Processing ---
if uploaded_files and job_description:
    with st.spinner("Processing Resumes..."):  # Show a loading indicator
        resumes = []
        resume_names = []
        valid_files = True  # Flag to track if all files were processed successfully
        for file in uploaded_files:
            text = extract_text_from_pdf(file)
            if text:
                resumes.append(text)
                resume_names.append(file.name)
            else:
                valid_files = False  # Mark as invalid if a file fails
                break  # Stop processing further files

        if valid_files:
            if resumes:  # Make sure we actually have resumes to process
                scores = rank_resumes(job_description, resumes)
                results = pd.DataFrame(
                    {"Resume": resume_names, "Score": scores}
                )  # Use the names gathered earlier
                results = results.sort_values(by="Score", ascending=False).reset_index(
                    drop=True
                )
                results.index += 1  # Start index at 1 for better readability

                st.markdown("<p class='medium-font'>Ranking Results</p>", unsafe_allow_html=True)
                st.dataframe(results, width=800, height=400) # Make dataframe more readable

                # Add a download option
                csv = results.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name="resume_ranking.csv",
                    mime="text/csv",
                )
            else:
                st.warning("No resumes were successfully processed. Please check the PDF files.")
        else:
            st.error("An error occurred during PDF processing.  Please check the file(s).")
elif uploaded_files and not job_description:
    st.warning("Please enter a job description to rank the resumes.")
elif job_description and not uploaded_files:
    st.warning("Please upload resumes to rank.")

# --- Footer ---
st.markdown("---")
st.markdown(
    "**Note:** This tool provides an initial screening and ranking based on text similarity.  It is essential to manually review the resumes and conduct interviews for a thorough evaluation."
)
st.markdown(
    "Developed by [Your Name/Organization].  For any questions or issues, please contact [Your Email/Contact]."
)
