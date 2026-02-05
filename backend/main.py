from fastapi import FastAPI, UploadFile, File, Form
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

@app.post("/analyze")
async def analyze_resume(
        resume: UploadFile = File(...),
        job_description: str = Form(...)
):
    resume_text = extract_text_from_pdf(resume.file)

    text = [resume_text, job_description]

    cv = CountVectorizer().fit_transform(text)
    similarity_score = cosine_similarity(cv)[0][1]

    ats_score = round(similarity_score * 100, 2)

    return {
        "ATS_score": ats_score
    }