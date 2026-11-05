# AI Resume Job Matcher

An AI-powered web application that analyzes how well a resume matches a job description and provides personalized recommendations to improve it.

Built with **Python, LangChain, LangGraph, OpenAI, and Streamlit**, the application uses AI to extract technical skills, compare job requirements, identify missing qualifications, and generate an ATS-style compatibility score.

The application is containerized using Docker and has been deployed to AWS using Amazon ECS and Amazon ECR.

## Features

- **PDF Resume Upload:** Upload a resume in PDF format.
- **Job Description Analysis:** Paste a job description to compare against the resume.
- **AI Skill Extraction:** Automatically identify technical skills from both the resume and job description.
- **ATS Match Score:** Generate an overall compatibility score from 0 to 100.
- **Detailed Scoring:** Evaluate technical skills, experience, education, and keyword coverage separately.
- **Skill Gap Analysis:** Identify direct matches, related experience, missing keywords, and unmet requirements.
- **Resume Recommendations:** Receive up to three actionable suggestions to improve the resume.
- **ATS Keyword Analysis:** Identify relevant keywords already included and keywords that could be emphasized.
- **Truthful Recommendations:** AI is instructed not to invent skills, experience, or qualifications.

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core application logic |
| Streamlit | Web interface |
| LangChain | OpenAI integration and prompt management |
| LangGraph | Multi-step AI workflow |
| OpenAI API (GPT-5-mini) | Resume and job description analysis |
| Pydantic | Structured AI responses |
| pypdf | PDF text extraction |
| Docker | Application containerization |
| Amazon ECR | Docker image storage |
| Amazon ECS | Cloud application deployment |

## How It Works

1. **Upload Resume:** The user uploads a PDF resume through the Streamlit interface.
2. **Enter Job Description:** The user pastes the job posting they want to analyze.
3. **Extract Skills:** LangGraph coordinates AI steps that identify important skills from the job description and resume.
4. **Analyze Compatibility:** OpenAI compares the extracted skills, experience, education, and keywords against the job requirements.
5. **Calculate Score:** Python calculates the final weighted match score.
6. **Display Results:** Streamlit presents the scores, matching skills, missing requirements, and recommended improvements.

### LangGraph Workflow

```text
START
  |
  v
Extract Job Skills
  |
  v
Extract Resume Skills
  |
  v
Analyze Resume
  |
  v
Calculate Weighted Score
  |
  v
Display Results
  |
  v
END
```

## ATS Match Scoring

The overall match score is calculated using four weighted categories:

| Category | Weight |
|---|---|
| Technical Skills | 35% |
| Experience | 30% |
| Education | 15% |
| Keyword Coverage | 20% |

**Formula:**

```text
Final Score =
    (Technical Match × 0.35)
  + (Experience Match × 0.30)
  + (Education Match × 0.15)
  + (Keyword Match × 0.20)
```

The score is an AI-generated estimate of resume compatibility, not an official score from an employer's Applicant Tracking System.

## Getting Started

### Prerequisites

- Python 3.14 or a compatible Python version
- An OpenAI API key
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/ayousef5/LangChainJobMatcher.git
cd LangChainJobMatcher
```

### 2. Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install streamlit langchain langchain-openai langgraph python-dotenv pypdf pydantic
```

### 4. Configure Environment Variables

Create a `.env` file in the project directory:

```env
OPENAI_API_KEY=your_openai_api_key
```

Do not commit your API key to GitHub.

### 5. Run the Application

```bash
streamlit run web_app.py
```

Open the local Streamlit address shown in your terminal, usually:

```text
http://localhost:8501
```

## Running with Docker

Build the Docker image:

```bash
docker build -t ai-resume-matcher .
```

Run the container:

```bash
docker run -p 8501:8501 \
  -e OPENAI_API_KEY=your_openai_api_key \
  ai-resume-matcher
```

Open `http://localhost:8501` in your browser.

## AWS Deployment

The application was containerized with Docker and deployed using AWS services.

- **Amazon ECR:** Stores the application's Docker image.
- **Amazon ECS:** Runs the containerized Streamlit application.
- **Environment Variables:** Provide API configuration without hardcoding credentials.

## Project Structure

```text
LangChainJobMatcher/
├── app.py           # Initial command-line resume matcher
├── web_app.py       # Streamlit app and LangGraph workflow
├── Dockerfile       # Docker container configuration
├── .dockerignore    # Files excluded from Docker builds
├── .gitignore       # Files excluded from Git
└── README.md        # Project documentation
```

## Future Improvements

- Support DOCX resume uploads.
- Allow users to compare multiple job descriptions.
- Export analysis results as a PDF report.
- Add resume analysis history.
- Improve score consistency through additional testing.

## Author

**Ahmed Yousef**

[GitHub](https://github.com/ayousef5) | [LinkedIn](https://www.linkedin.com/in/ahmedyousef1/)
