# AI Resume Analyzer

---

## 📌 Project Overview

This project is a web application that allows users to upload a resume (PDF) and receive AI-generated structured feedback.

The system analyzes the resume and provides:

* Overall score (out of 100)
* Profile summary
* Key strengths
* Areas for improvement
* Missing skills / sections
* Suggestions for improvement

### Additional Features

* Job description matching
* Section-wise scoring (/10)
* Keywords extraction
* Download analysis as PDF

---

## 🎯 Objective

The objective of this project is to build an AI-powered Resume Analyzer.

This includes:

* Using an AI model (Groq LLM) to analyze resume content and generate structured feedback
* Designing prompts to guide the AI in producing consistent and useful output
* Validating and correcting AI-generated responses (handling formatting and JSON issues)
* Building a complete working application within a limited timeframe
* Making practical decisions in integrating AI with a web application

---

## ⚙️ Tech Stack

### Frontend

* HTML
* CSS
* JavaScript

### Backend

* Python (Flask)

### AI

* Groq API (LLaMA model)

### Libraries / Tools

* pdfplumber (PDF text extraction)
* spaCy (NLP processing)
* reportlab (PDF generation)

---

## ⚙️ How to Run the Project

### 1. Create Virtual Environment

```bash
py -3.11 -m venv .venv
.\.venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r backend/requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Add API Key

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key_here
```

### 4. Run the Application

```bash
python backend/app.py
```

### 5. Open in Browser

```
http://localhost:5000
```

---

## 🧠 Architecture / Flow

1. User uploads resume (PDF)
2. Backend extracts text using `pdfplumber`
3. User optionally enters Job Description
4. Backend sends prompt to AI (Groq API)
5. AI returns structured JSON response
6. Backend cleans and parses the response
7. Frontend displays results
8. User can download analysis as PDF

---

## 🤖 AI Tools Used

* **Groq API (LLaMA model)**

---

## 🧾 Prompt Used

The AI is instructed to return structured JSON including:

```json
{
  "overall_score": number,
  "job_match_score": number,
  "profile_summary": "...",
  "section_scores": {},
  "section_feedback": {},
  "key_strengths": [],
  "areas_for_improvement": [],
  "missing_skills": [],
  "suggestions": [],
  "keywords_found": []
}
```

---

## ✅ Validation of AI Output

### Issues:

* AI sometimes returned output inside ```json block
* Sometimes JSON format was incorrect

### Fix:

* Cleaned response using string replacement
* Parsed using `json.loads()`
* Added fallback handling

---

## 👍 What Worked Well

* Structured prompt improved consistency
* Clean UI for displaying results
* Job description matching improved accuracy
* Section-wise scoring added better insights

---

## ⚠️ Where AI Output Needed Fix

* Inconsistent JSON formatting
* Missing fields occasionally

### Solution:

* Added validation logic
* Cleaned AI response before parsing

---

## 📌 Assumptions

* Resume is provided in readable PDF format
* AI returns structured output
* Job description is optional

---

## ❌ Limitations

* AI output may vary slightly
* Works best with well-formatted resumes
* Limited support for DOC/DOCX

---

## 🚀 Future Improvements

* Improve error handling
* Highlight missing keywords
* Add score visualization (charts/progress bars)
* Deploy application online

---

## 🧠 AI Usage Declaration

### AI was used for:

* Prompt design
* Debugging backend issues
* Improving UI

### Manual work included:

* Fixing JSON parsing issues
* Validating AI outputs
* Debugging frontend rendering

---
