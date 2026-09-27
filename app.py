import os
import io
import uvicorn

from typing import TypedDict

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse

from langserve import add_routes

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, END

from pypdf import PdfReader


# ============================================================
# 1. API KEY
# ============================================================

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set.")


# ============================================================
# 2. LLM
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)


# ============================================================
# 3. LANGGRAPH STATE
# ============================================================

class CareerState(TypedDict, total=False):
    question: str
    resume: str
    role: str
    github_id: str

    job_results: str
    skill_gap: str
    project_suggestions: str
    github_analysis: str
    final_answer: str


# ============================================================
# 4. JOB SEARCH NODE
# ============================================================

def job_search(state: CareerState):

    prompt = ChatPromptTemplate.from_template(
        """
        You are an AI career assistant.

        Student resume:
        {resume}

        Target role:
        {role}

        Find suitable job-role recommendations based on the
        student's skills and target role.

        Give a short answer with:

        1. Suitable job roles
        2. Important technologies
        3. What the student should prepare

        Do not invent specific companies or job openings.
        """
    )

    chain = prompt | llm

    response = chain.invoke({
        "resume": state["resume"],
        "role": state["role"]
    })

    return {
        "job_results": response.content
    }


# ============================================================
# 5. SKILL GAP NODE
# ============================================================

def skill_gap(state: CareerState):

    prompt = ChatPromptTemplate.from_template(
        """
        Analyze this student's resume for the target role.

        RESUME:
        {resume}

        TARGET ROLE:
        {role}

        Identify:

        1. Existing strengths
        2. Missing skills
        3. Skills that should be prioritized
        4. Interview preparation areas

        Keep the answer concise.
        """
    )

    chain = prompt | llm

    response = chain.invoke({
        "resume": state["resume"],
        "role": state["role"]
    })

    return {
        "skill_gap": response.content
    }


# ============================================================
# 6. PROJECT SUGGESTIONS NODE
# ============================================================

def project_suggestions(state: CareerState):

    prompt = ChatPromptTemplate.from_template(
        """
        Suggest practical projects for a student who wants to
        become an {role}.

        STUDENT RESUME:
        {resume}

        Suggest 3 projects.

        For each project provide:

        - Project name
        - Short description
        - Technologies
        - Why it helps for placement

        Prefer AI/ML/Generative AI projects when appropriate.
        """
    )

    chain = prompt | llm

    response = chain.invoke({
        "resume": state["resume"],
        "role": state["role"]
    })

    return {
        "project_suggestions": response.content
    }


# ============================================================
# 7. GITHUB CHECK NODE
# ============================================================

def github_check(state: CareerState):

    prompt = ChatPromptTemplate.from_template(
        """
        Analyze the student's GitHub information.

        GitHub username:
        {github_id}

        Resume:
        {resume}

        Important:

        You cannot access the GitHub account directly.

        Therefore, do NOT invent repository names, commits,
        stars, followers, or activity.

        Instead provide:

        - What should be checked in the GitHub profile
        - What makes a GitHub profile placement-ready
        - Suggestions to improve the profile
        """
    )

    chain = prompt | llm

    response = chain.invoke({
        "github_id": state["github_id"],
        "resume": state["resume"]
    })

    return {
        "github_analysis": response.content
    }


# ============================================================
# 8. FINAL SYNTHESIS NODE
# ============================================================

def final_synthesis(state: CareerState):

    prompt = ChatPromptTemplate.from_template(
        """
        You are a placement-ready AI career advisor.

        Combine the following analysis into one clear final answer.

        JOB RECOMMENDATIONS:
        {job_results}

        SKILL GAP:
        {skill_gap}

        PROJECT SUGGESTIONS:
        {project_suggestions}

        GITHUB ANALYSIS:
        {github_analysis}

        STUDENT QUESTION:
        {question}

        Give a concise final career plan.

        Include:

        1. Current position
        2. Main skill gaps
        3. Recommended projects
        4. Job preparation plan
        5. Final advice

        Do not mention internal nodes or LangGraph.
        """
    )

    chain = prompt | llm

    response = chain.invoke({
        "job_results": state["job_results"],
        "skill_gap": state["skill_gap"],
        "project_suggestions": state["project_suggestions"],
        "github_analysis": state["github_analysis"],
        "question": state["question"]
    })

    return {
        "final_answer": response.content
    }


# ============================================================
# 9. BUILD LANGGRAPH
# ============================================================

builder = StateGraph(CareerState)

builder.add_node("job_search", job_search)
builder.add_node("skill_gap", skill_gap)
builder.add_node("project_suggestions", project_suggestions)
builder.add_node("github_check", github_check)
builder.add_node("final_synthesis", final_synthesis)

builder.set_entry_point("job_search")

builder.add_edge("job_search", "skill_gap")
builder.add_edge("skill_gap", "project_suggestions")
builder.add_edge("project_suggestions", "github_check")
builder.add_edge("github_check", "final_synthesis")
builder.add_edge("final_synthesis", END)

graph = builder.compile()


# ============================================================
# 10. LANGSERVE INPUT
# ============================================================

class CareerAgentInput(BaseModel):
    question: str = Field(
        description="The student's career question"
    )

    resume: str = Field(
        description="Extracted resume text"
    )

    role: str = Field(
        description="The target job role"
    )

    github_id: str = Field(
        description="The student's GitHub username"
    )


def extract_final_answer(graph_output: dict) -> str:

    if isinstance(graph_output, dict):
        return graph_output.get(
            "final_answer",
            str(graph_output)
        )

    return str(graph_output)


formatted_graph_chain = (
    graph
    | RunnableLambda(extract_final_answer)
).with_types(
    input_type=CareerAgentInput,
    output_type=str
)


# ============================================================
# 11. FASTAPI APP
# ============================================================

app = FastAPI(
    title="Placement Ready AI Career Agent",
    version="2.0"
)


# ============================================================
# 12. LANGSERVE PLAYGROUND
# ============================================================

add_routes(
    app,
    formatted_graph_chain,
    path="/agent",
    playground_type="default"
)


# ============================================================
# 13. PDF EXTRACTION
# ============================================================

def extract_pdf_text(pdf_bytes: bytes) -> str:

    try:

        pdf_file = io.BytesIO(pdf_bytes)

        reader = PdfReader(pdf_file)

        extracted_text = ""

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                extracted_text += page_text + "\n"

        return extracted_text.strip()

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=f"Could not read PDF: {str(e)}"
        )


# ============================================================
# 14. RESUME ANALYSIS API
# ============================================================

@app.post("/analyze")
async def analyze_resume(
    resume: UploadFile = File(...),
    role: str = Form(...),
    github_id: str = Form(...),
    question: str = Form(...)
):

    if not resume.filename:
        raise HTTPException(
            status_code=400,
            detail="Please select a resume."
        )

    if not resume.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF resumes are supported."
        )

    pdf_bytes = await resume.read()

    if not pdf_bytes:
        raise HTTPException(
            status_code=400,
            detail="The uploaded PDF is empty."
        )

    resume_text = extract_pdf_text(pdf_bytes)

    if not resume_text:
        raise HTTPException(
            status_code=400,
            detail=(
                "No readable text was found in the PDF. "
                "Please upload a text-based PDF."
            )
        )

    result = graph.invoke({
        "question": question,
        "resume": resume_text,
        "role": role,
        "github_id": github_id
    })

    return {
        "success": True,
        "filename": resume.filename,
        "extracted_characters": len(resume_text),
        "final_answer": result.get(
            "final_answer",
            "No final answer generated."
        )
    }


# ============================================================
# 15. FRONTEND
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def home():

    return """
<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<meta name="viewport"
      content="width=device-width, initial-scale=1.0">

<title>Placement Ready AI</title>

<style>

* {
    box-sizing: border-box;
}

body {

    margin: 0;

    font-family: Arial, sans-serif;

    background:
        linear-gradient(
            135deg,
            #0f172a,
            #1e293b
        );

    min-height: 100vh;

    color: white;

    padding: 40px 20px;
}

.container {

    max-width: 900px;

    margin: auto;
}

.header {

    text-align: center;

    margin-bottom: 30px;
}

.header h1 {

    font-size: 42px;

    margin-bottom: 10px;
}

.header p {

    color: #cbd5e1;

    font-size: 17px;
}

.card {

    background: white;

    color: #0f172a;

    border-radius: 20px;

    padding: 30px;

    box-shadow:
        0 20px 60px
        rgba(0, 0, 0, 0.3);
}

.form-group {

    margin-bottom: 22px;
}

label {

    display: block;

    font-weight: bold;

    margin-bottom: 8px;
}

input,
textarea {

    width: 100%;

    padding: 13px;

    border: 1px solid #cbd5e1;

    border-radius: 10px;

    font-size: 15px;
}

.file-box {

    border: 2px dashed #94a3b8;

    border-radius: 14px;

    padding: 25px;

    text-align: center;

    background: #f8fafc;
}

button {

    width: 100%;

    padding: 15px;

    border: none;

    border-radius: 11px;

    background: #4f46e5;

    color: white;

    font-size: 17px;

    font-weight: bold;

    cursor: pointer;
}

button:hover {

    background: #4338ca;
}

button:disabled {

    background: #94a3b8;

    cursor: not-allowed;
}

.loading {

    display: none;

    text-align: center;

    margin-top: 20px;

    color: #475569;
}

.result {

    display: none;

    margin-top: 30px;
}

.result-box {

    background: #f8fafc;

    border: 1px solid #e2e8f0;

    border-radius: 14px;

    padding: 25px;

    line-height: 1.7;

    white-space: pre-wrap;
}

.success {

    background: #ecfdf5;

    border: 1px solid #a7f3d0;

    color: #065f46;

    padding: 12px;

    border-radius: 10px;

    margin-bottom: 15px;
}

.error {

    display: none;

    background: #fef2f2;

    border: 1px solid #fecaca;

    color: #991b1b;

    padding: 12px;

    border-radius: 10px;

    margin-top: 20px;
}

.footer {

    text-align: center;

    margin-top: 25px;

    color: #94a3b8;

    font-size: 13px;
}

</style>

</head>


<body>

<div class="container">

<div class="header">

<h1>🤖 Placement Ready AI</h1>

<p>
Upload your resume and get an AI-powered career analysis.
</p>

</div>


<div class="card">

<form id="resumeForm">


<div class="form-group">

<label>📄 Upload Resume PDF</label>

<div class="file-box">

<input
    type="file"
    id="resume"
    name="resume"
    accept=".pdf"
    required
>

<p>
Select your resume PDF
</p>

</div>

</div>


<div class="form-group">

<label>💼 Target Job Role</label>

<input
    type="text"
    id="role"
    name="role"
    placeholder="Example: AI Engineer"
    required
>

</div>


<div class="form-group">

<label>🔗 GitHub Username</label>

<input
    type="text"
    id="github_id"
    name="github_id"
    placeholder="Example: Bharath-max143"
    required
>

</div>


<div class="form-group">

<label>💬 Your Question</label>

<textarea
    id="question"
    name="question"
    rows="4"
    placeholder="Example: Am I ready for an AI Engineer role?"
    required
></textarea>

</div>


<button
    type="submit"
    id="analyzeBtn"
>
🚀 Analyze Resume
</button>


</form>


<div
    class="loading"
    id="loading"
>

⏳ Analyzing your resume...

<br>

Please wait while the AI agents work.

</div>


<div
    class="result"
    id="result"
>

<h2>📊 Career Readiness Report</h2>

<div
    class="success"
    id="successMessage"
>
</div>

<div
    class="result-box"
    id="resultBox"
>
</div>

</div>


<div
    class="error"
    id="errorBox"
>
</div>


</div>


<div class="footer">

Placement Ready AI • LangChain + LangGraph + FastAPI

</div>

</div>


<script>

const form =
    document.getElementById("resumeForm");

const button =
    document.getElementById("analyzeBtn");

const loading =
    document.getElementById("loading");

const result =
    document.getElementById("result");

const resultBox =
    document.getElementById("resultBox");

const successMessage =
    document.getElementById("successMessage");

const errorBox =
    document.getElementById("errorBox");


form.addEventListener(
    "submit",
    async function(event) {

        event.preventDefault();

        result.style.display = "none";

        errorBox.style.display = "none";

        loading.style.display = "block";

        button.disabled = true;

        button.innerText = "⏳ Analyzing...";


        const file =
            document.getElementById(
                "resume"
            ).files[0];

        const role =
            document.getElementById(
                "role"
            ).value;

        const github =
            document.getElementById(
                "github_id"
            ).value;

        const question =
            document.getElementById(
                "question"
            ).value;


        const formData =
            new FormData();


        formData.append(
            "resume",
            file
        );

        formData.append(
            "role",
            role
        );

        formData.append(
            "github_id",
            github
        );

        formData.append(
            "question",
            question
        );


        try {

            const response =
                await fetch(
                    "/analyze",
                    {
                        method: "POST",
                        body: formData
                    }
                );


            const data =
                await response.json();


            if (!response.ok) {

                throw new Error(
                    data.detail ||
                    "Analysis failed."
                );

            }


            successMessage.innerText =
                "Resume analyzed successfully. " +
                "Extracted " +
                data.extracted_characters +
                " characters from " +
                data.filename;


            resultBox.innerText =
                data.final_answer;


            result.style.display =
                "block";


        } catch (error) {

            errorBox.innerText =
                "❌ " + error.message;

            errorBox.style.display =
                "block";

        }


        loading.style.display =
            "none";

        button.disabled =
            false;

        button.innerText =
            "🚀 Analyze Resume";

    }

);

</script>

</body>

</html>
"""


# ============================================================
# 16. RUN
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            8000
        )
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
