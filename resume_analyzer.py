import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2

nlp = spacy.load("en_core_web_lg")

class ResumeAnalyzer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words="english")

    def extract_text(self, pdf_path):
        """Extract text from PDF resumes"""
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            return " ".join([page.extract_text() for page in reader.pages])

    def preprocess(self, text):
        """Clean text using spaCy"""
        doc = nlp(text)
        return " ".join([
            token.lemma_.lower() 
            for token in doc 
            if not token.is_stop and not token.is_punct
        ])

    def rank_resumes(self, job_desc, resume_texts):
        """Calculate match scores using TF-IDF"""
        processed_job = self.preprocess(job_desc)
        processed_resumes = [self.preprocess(text) for text in resume_texts]
        
        tfidf_matrix = self.vectorizer.fit_transform([processed_job] + processed_resumes)
        scores = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1:])[0]
        
        return scores
