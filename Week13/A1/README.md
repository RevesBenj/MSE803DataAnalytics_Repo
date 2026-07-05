# AI CV Feedback and Optimization Backend

This backend application reads a user's CV and a target job description, generates personalized CV feedback, then uses that feedback to rewrite and optimize the user's actual CV. It exports feedback reports, JSON analysis, optimized DOCX, optimized PDF, and logs.

## Main Features

- Reads CV and job description from PDF, DOCX, or TXT.
- Performs ATS scoring with keyword, skills, experience, formatting, readability, and section scores.
- Detects missing CV sections such as summary, skills, projects, certifications, LinkedIn, GitHub, and portfolio.
- Generates recruiter-style feedback using Gemini or OpenAI when an API key is available.
- Uses a rule-based fallback when no API key is provided.
- Performs a second rewrite step to optimize the user's actual CV using the generated feedback.
- Exports:
  - `cv_feedback_report.md`
  - `cv_feedback_report.txt`
  - `analysis.json`
  - `optimized_cv.docx`
  - `optimized_cv.pdf`
  - `logs/analysis.log`

## Workflow

```text
User CV
  -> Read CV text
  -> Read job description text
  -> ATS and structure analysis
  -> Generate personalized feedback
  -> Rewrite the actual CV using the feedback
  -> Export DOCX, PDF, Markdown, TXT, JSON, and logs
```

## Installation

```bash
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Mac/Linux
pip install -r requirements.txt
```

## Optional API Keys

Gemini:

```bash
set LLM_PROVIDER=gemini
set GEMINI_API_KEY=your_key_here
```

OpenAI:

```bash
set LLM_PROVIDER=openai
set OPENAI_API_KEY=your_key_here
```

If no API key is added, the application still runs using the rule-based fallback.

## Run

```bash
python main.py --cv resume.pdf --jd job_description.pdf --output output
```

Specific exports:

```bash
python main.py --cv resume.pdf --jd job_description.pdf --provider gemini --output output --docx --pdf --markdown --json
```

## Assignment Requirement Coverage

| Requirement | Status |
|---|---|
| Generate constructive personalized feedback | Done |
| Recommend content, structure, and presentation improvements | Done |
| Use generated feedback to optimize the CV | Done |
| Generate updated CV in DOCX | Done |
| Generate updated CV in PDF | Done |
| Provide backend source code | Done |
| Prepare GitHub-ready project documentation | Done |

## Important Ethics Rule

The application is designed not to invent fake jobs, companies, certifications, dates, metrics, or experience. If evidence is missing, it uses wording such as `Add evidence if true` or `Add measurable result if available`.
