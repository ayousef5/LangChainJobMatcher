import streamlit as st  # Lets us build the website
st.set_page_config(
    page_title="AI Resume Job Matcher",  # Text shown in the browser tab
    page_icon="📄",  # Icon shown in the browser tab
    layout="centered"  # Keep the content centered
)
from pypdf import PdfReader  # Lets Python read PDF files
from dotenv import load_dotenv  # Loads our API key from .env
from langchain_openai import ChatOpenAI  # Lets us use OpenAI through LangChain
from langchain_core.prompts import ChatPromptTemplate  # Lets us create prompts
from pydantic import BaseModel, Field  # Lets us define the AI response structure
from langgraph.graph import StateGraph, START, END  # Tools for building a workflow
from typing import TypedDict  # Lets us define the structure of our workflow state

class ResumeState(TypedDict):
    resume: str  # Stores the resume text
    job_description: str  # Stores the job description
    job_skills: list[str]  # Stores important skills from the job
    resume_skills: list[str]  # Stores important skills from the resume
    result: object  # Stores the final AI analysis

load_dotenv()  # Load environment variables from .env file
def extract_job_skills(state: ResumeState):
    # Ask the AI to extract important skills from the job description
    response = skill_model.invoke(
        f"""
        Extract the most important technical skills and technologies
        from this job description.

        Job Description:
        {state["job_description"]}
        """
    )

    # Save the list of skills in the workflow state
    return {
        "job_skills": response.skills
    }

def extract_resume_skills(state: ResumeState):
    # Ask the AI to find important skills in the resume
    response = skill_model.invoke(
        f"""
        Extract the most important technical skills and technologies
        from this resume.

        Resume:
        {state["resume"]}
        """
    )

    # Save the skills in the workflow state
    return {
        "resume_skills": response.skills
    }

def analyze_resume(state: ResumeState):
    # Print the skills extracted by the previous node

    # Send the resume, job description, and extracted skills to the AI
    response = chain.invoke({
        "resume": state["resume"],
        "job_description": state["job_description"],
        "job_skills": state["job_skills"],
        "resume_skills": state["resume_skills"]
    })

    # Calculate the final ATS score using fixed weights
    response.match_score = round(
        response.technical_match * 0.35
        + response.experience_match * 0.30
        + response.education_match * 0.15
        + response.keyword_match * 0.20
    )

    # Return the AI analysis
    return {
        "result": response
    }
# Create a graph using our ResumeState structure
graph = StateGraph(ResumeState)

# Add the job skill extraction node
graph.add_node("extract_job_skills", extract_job_skills)

# Add the resume skill extraction node
graph.add_node("extract_resume_skills", extract_resume_skills)

# Add the resume analysis node
graph.add_node("analyze_resume", analyze_resume)

# Start the workflow by extracting job skills
graph.add_edge(START, "extract_job_skills")

# Analyze the resume after extracting both sets of skills
graph.add_edge("extract_job_skills", "extract_resume_skills")
graph.add_edge("extract_resume_skills", "analyze_resume")

# End the workflow after the resume analysis
graph.add_edge("analyze_resume", END)

# Compile the graph so we can run it
workflow = graph.compile()


class JobMatch(BaseModel):
    match_score: int = Field(
        default=0,
        description="Final ATS score calculated by Python"
    )

    technical_match: int = Field(
        description="Score from 0 to 100 for how well the resume matches the job's technical skills"
    )

    experience_match: int = Field(
        description="Score from 0 to 100 for how well the resume matches the job's experience requirements"
    )

    education_match: int = Field(
        description="Score from 0 to 100 for how well the resume matches the education and eligibility requirements"
    )

    keyword_match: int = Field(
        description="Score from 0 to 100 for how well the resume covers important ATS keywords from the job"
    )

    direct_matches: list[str] = Field(
        description="Important job skills and keywords clearly demonstrated by the resume"
    )

    related_experience: list[str] = Field(
        description="Relevant resume experience related to a job requirement but not an exact match"
    )

    missing_keywords: list[str] = Field(
        description="Important job technologies, tools, or concepts not demonstrated in the resume"
    )

    requirement_gaps: list[str] = Field(
        description="Requirements the resume does not currently satisfy"
    )

    resume_improvements: list[str] = Field(
        min_length=0,
        max_length=3,
        description="At most 3 specific and truthful resume changes supported by the existing resume"
    )

    ats_keywords_present: list[str] = Field(
        description="Important job keywords already clearly represented in the resume"
    )

    ats_keywords_to_emphasize: list[str] = Field(
        description="Important job keywords supported by the resume but not stated clearly enough"
    )


class JobSkills(BaseModel):
    skills: list[str] = Field(
        description="Important technical skills and technologies from the job description"
    )

skill_model = ChatOpenAI(
    model="gpt-5-mini"  # Use the OpenAI model
).with_structured_output(JobSkills)  # Ask for a list of skills

model = ChatOpenAI(
    model="gpt-5-mini"  # Choose the OpenAI model
).with_structured_output(JobMatch)  # Tell LangChain what format we want back

prompt = ChatPromptTemplate.from_template(
    """
    Analyze this resume against this job description with the goal of improving
    ATS keyword matching while staying completely truthful.

    Resume:
    {resume}

    Job Description:
    {job_description}

    Important skills extracted from the job:
    {job_skills}

    Important skills extracted from the resume:
    {resume_skills}

    Compare the resume against the job requirements.

    Give me:

    1. Direct Matches
       - Important job skills and keywords that are clearly demonstrated
         by the resume.

    2. Related Experience
       - Resume experience that is relevant to a job requirement but does
         not directly match the specific technology or terminology.
       - Explain why it is related.

    3. Missing Keywords
       - Important technologies, tools, or concepts from the job description
         that are not demonstrated anywhere in the resume.

    4. Requirement Gaps
       - Requirements that the resume does not currently satisfy, such as
         minimum experience, GPA, degree, work authorization, or required
         duration of experience.
       - Explain the specific reason for each gap.

    5. ATS Match Score
       - Evaluate these four categories separately:
         - Technical Match
         - Experience Match
         - Education Match
         - Keyword Match
       - Give each category a score from 0 to 100.
       - Do not let related experience count as direct experience.
       - Required qualifications should have more importance than minor keywords.
       - Missing required experience or eligibility requirements should significantly
         reduce the relevant score.
       - The final match score should reflect the overall strength of the candidate's
         match, not simply the number of matching keywords.

    6. Resume Improvements
       - Suggest specific changes to the existing resume that could improve
         ATS matching.
       - Every suggestion must be directly supported by evidence already present
         in the resume.
       - You may suggest making an existing skill, technology, accomplishment,
         duration, or responsibility more explicit.
       - Do not suggest adding a skill, technology, qualification, interest,
         experience, or keyword that is not supported by the resume.
       - Do not suggest adding technologies simply because they appear in the
         job description.
       - If an important job requirement is genuinely missing from the resume,
         report it as a Missing Keyword or Requirement Gap instead of suggesting
         that the candidate add it.
       - Prioritize the 3 most useful and truthful changes.

    7. ATS Keyword Coverage
       - List important job keywords that are already clearly represented
         in the resume.
       - List important keywords that are supported by the resume but could
         be stated more explicitly.

    Rules:
    - Never invent skills, experience, technologies, accomplishments, or
      qualifications.
    - Do not keyword stuff.
    - Never call a different technology an equivalent or substitute for the
      required technology.
    - Related experience may explain why the candidate's experience is relevant,
      but it must never imply experience with a technology that is not listed
      in the resume.
    - Do not suggest adding a keyword to the resume unless the resume provides
      direct evidence that the candidate has actually used or learned it.
    - Do not infer experience with Ansible, Terraform, Chef, Puppet, VMware,
      Infrastructure as Code, machine provisioning, or similar technologies
      from general automation, cloud, API, or pipeline experience.
    - Distinguish between having relevant experience and having the exact
      required technology.
    - If a requirement cannot be determined from the resume, say that it
      cannot be determined rather than assuming the candidate has it.
    """
)
chain = prompt | model  # Connect the prompt to the AI model

st.title("AI Resume Job Matcher")  # Shows the title on the page

st.write("Upload your resume and paste a job description to see how well they match.")  # Shows instructions
uploaded_resume = st.file_uploader(
    "Upload your resume",  # Text shown above the upload button
    type="pdf"  # Only allow PDF files
)

job_description = st.text_area(
    "Paste the job description",  # Label shown above the text box
    height=250  # Make the text box taller
)

analyze_button = st.button(
    "Analyze Resume"  # Create the analyze button
)

if uploaded_resume and job_description:  # Continue only if both inputs exist # Check if a resume was uploaded
    reader = PdfReader(uploaded_resume)  # Open the uploaded PDF

    resume_text = ""  # Create a place to store the resume text

    for page in reader.pages:  # Go through each page
        resume_text += page.extract_text()  # Extract the page text

    if analyze_button:  # Run the AI when the button is clicked
        with st.spinner("Analyzing your resume..."):  # Show a loading message
            # Create the starting state for LangGraph
            state = {
                "resume": resume_text,
                "job_description": job_description
            }

            # Run the LangGraph workflow
            result = workflow.invoke(state)

            # Get the AI response from the workflow
            response = result["result"]

        st.subheader("ATS Match Score")

        st.metric("Overall Score", f"{response.match_score}/100")

        st.progress(response.match_score / 100)

        st.write(f"**Technical Match:** {response.technical_match}/100")
        st.progress(response.technical_match / 100)

        st.write(f"**Experience Match:** {response.experience_match}/100")
        st.progress(response.experience_match / 100)

        st.write(f"**Education Match:** {response.education_match}/100")
        st.progress(response.education_match / 100)

        st.write(f"**Keyword Match:** {response.keyword_match}/100")
        st.progress(response.keyword_match / 100)

        st.subheader("🟢 Direct Matches")
        for item in response.direct_matches:
            st.write(f"✅ {item}")

        st.subheader("🟡 Related Experience")
        for item in response.related_experience:
            st.write(f"🔹 {item}")

        st.subheader("🔴 Missing Keywords")
        for item in response.missing_keywords:
            st.write(f"❌ {item}")

        st.subheader("⚠️ Requirement Gaps")
        for item in response.requirement_gaps:
            st.write(f"⚠️ {item}")

        st.subheader("💡 Resume Improvements")
        for item in response.resume_improvements:
            st.write(f"💡 {item}")

        st.subheader("🔎 ATS Keywords Already Present")
        for item in response.ats_keywords_present:
            st.write(f"✅ {item}")

        st.subheader("📝 ATS Keywords to Emphasize")
        for item in response.ats_keywords_to_emphasize:
            st.write(f"📌 {item}")

if not uploaded_resume:  # Check if no resume was uploaded
    st.info("Please upload your resume.")  # Tell the user what to do

if not job_description:  # Check if no job description was entered
    st.info("Please paste a job description.")  # Tell the user what to do


