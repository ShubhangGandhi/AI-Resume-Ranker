from flask import Flask, render_template, request, redirect, url_for
from resume_analyzer import ResumeAnalyzer
import os

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

analyzer = ResumeAnalyzer()

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Get job description and resumes
        job_desc = request.form["job_description"]
        uploaded_files = request.files.getlist("resumes")
        
        # Process files
        resume_texts = []
        filenames = []
        for file in uploaded_files:
            if file.filename.endswith(".pdf"):
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
                file.save(filepath)
                resume_texts.append(analyzer.extract_text(filepath))
                filenames.append(file.filename)
        
        # Rank resumes
        scores = analyzer.rank_resumes(job_desc, resume_texts)
        results = list(zip(filenames, scores))
        results.sort(key=lambda x: x[1], reverse=True)  # Highest score first
        
        return render_template("results.html", rankings=results)
    
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
