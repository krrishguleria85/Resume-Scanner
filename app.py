# %%writefile app.py
import streamlit as st
import docx
from pdfminer.high_level import extract_text
import re
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --- Helper Functions ---
def extract_text_from_pdf(pdf_file):
    return extract_text(pdf_file)

def extract_text_from_docx(docx_file):
    doc = docx.Document(docx_file)
    full_text = [para.text for para in doc.paragraphs]
    return '\n'.join(full_text)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def calculate_skill_match(resume_text, job_description_text):
    if not resume_text or not job_description_text:
        return 0.0, []

    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform([resume_text, job_description_text])
    
    # Calculate cosine similarity
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0] * 100

    # Identify missing skills
    resume_words = set(resume_text.split())
    job_words = set(job_description_text.split())
    missing_skills = sorted(list(job_words - resume_words))

    return similarity, missing_skills

# --- Streamlit UI ---
st.set_page_config(page_title="Resume-Job Description Matcher", layout="centered")
st.title(" Resume-Job Description Matcher")
st.markdown("""
    Upload your resume and the job description to get a **skill match score** and identify any **missing keywords**.
""")
# File Uploaders
st.subheader("Upload Documents")

resume_file = st.file_uploader("Upload your Resume (PDF or DOCX)", type=["pdf", "docx"])
job_description_file = st.file_uploader("Upload the Job Description (PDF or DOCX)", type=["pdf", "docx"])

resume_raw_text = ""
job_description_raw_text = ""

if resume_file:
    with st.spinner("Processing Resume..."):
        if resume_file.type == "application/pdf":
            resume_raw_text = extract_text_from_pdf(resume_file)
        elif resume_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            resume_raw_text = extract_text_from_docx(resume_file)
        else:
            st.error("Unsupported resume file type. Please upload a PDF or DOCX.")

if job_description_file:
    with st.spinner("Processing Job Description..."):
        if job_description_file.type == "application/pdf":
            job_description_raw_text = extract_text_from_pdf(job_description_file)
        elif job_description_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            job_description_raw_text = extract_text_from_docx(job_description_file)
        else:
            st.error("Unsupported job description file type. Please upload a PDF or DOCX.")

if st.button("Analyze Match"):
    if resume_raw_text and job_description_raw_text:
        st.subheader("Analysis Results")

        # Clean texts
        cleaned_resume_text = clean_text(resume_raw_text)
        cleaned_job_description_text = clean_text(job_description_raw_text)

        # Calculate match
        match_score, missing_skills = calculate_skill_match(cleaned_resume_text, cleaned_job_description_text)
        st.markdown(f"### Skill Match Score: **<span style='color:green;'>{match_score:.2f}%</span>**", unsafe_allow_html=True)
        st.write("---")
        st.subheader("Missing Skills/Keywords")
        if missing_skills:
            st.warning("The following keywords from the job description were **not found** in your resume:")
            st.markdown(f"**`{', '.join(missing_skills)}`**")
            st.info("Consider adding these relevant keywords to your resume to improve your match!")
        else:
            st.success("Great! No significant missing skills found based on the job description keywords.")
            st.info("Your resume appears to cover the key terms from the job description well.")
    else:
        st.warning("Please upload both your resume and the job description to perform the analysis.")
st.markdown("---")
st.markdown("*This tool provides an automated keyword matching score and highlights potential gaps. Always review your resume manually for a perfect fit!*")