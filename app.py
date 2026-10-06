from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field  # Lets us define the structure of the AI response
from pypdf import PdfReader  # Lets Python read PDF files
load_dotenv()

reader = PdfReader("resume.pdf")  # Open resume pdf
resume_text = "" # Create string to hold resume text
for page in reader.pages:
    resume_text += page.extract_text()  # Extract text from each page of the PDF

class JobMatch(BaseModel):
    match_score: int = Field(description="Match score from 0 to 100")

    matching_skills: list[str] = Field(description="Skills that match the job")

    missing_skills: list[str] = Field(description="Skills that are missing from the resume")
    
    improvements: list[str] = Field(description="Specific resume improvements to better match the job")

model = ChatOpenAI(
    model="gpt-5-mini"
).with_structured_output(JobMatch)

prompt = ChatPromptTemplate.from_template(
    """
    Compare this resume to this job description.

    Resume:
    {resume}

    Job Description:
    {job_description}

    Give me:
    1. Matching skills
    2. Missing skills
    3. A match score from 0 to 100
    """
)
chain = prompt | model # Connect the prompt to the AI model

job_description = """
Looking for a software engineering intern with Python, C++, SQL,
and experience building web applications.
"""  # Store the job description in a variable

response = chain.invoke({
    "resume": resume_text,  # Send the resume text to the AI
    "job_description": job_description  # Send the job description to the AI
})

print(response)  # Print the AI's structured response