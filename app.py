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
# GOOGLE API KEY
# ============================================================

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set.")


# ============================================================
# LLM
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)


# ============================================================
# HELPER - CONVERT AI RESPONSE TO TEXT
# ============================================================

def content_to_text(content):
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, str):
                parts.append(item)

            elif isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))
                elif "content" in item:
                    parts.append(str(item["content"]))

            elif hasattr(item, "text"):
                parts.append(str(item.text))

        return "\n".join(parts)

    if isinstance(content, dict):
        if "text" in content:
            return str(content["text"])

        if "content" in content:
            return str(content["content"])

    return str(content)


# ============================================================
# LANGGRAPH STATE
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
# NODE 1 - JOB SEARCH / RECOMMENDATIONS
# ============================================================

def job_search(state: CareerState):

    prompt = ChatPromptTemplate.from_template("""
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
""")

    chain = prompt | llm

    response = chain.invoke({
        "resume": state["resume"],
        "role": state["role"]
    })

    return {
        "job_results": content_to_text(response.content)
    }


# ============================================================
# NODE 2 - SKILL GAP
# ============================================================

def skill_gap(state: CareerState):

    prompt = ChatPromptTemplate.from_template("""
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
""")

    chain = prompt | llm

    response = chain.invoke({
        "resume": state["resume"],
        "role": state["role"]
    })

    return {
        "skill_gap": content_to_text(response.content)
    }


# ============================================================
# NODE 3 - PROJECT SUGGESTIONS
# ============================================================

def project_suggestions(state: CareerState):

    prompt = ChatPromptTemplate.from_template("""
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
""")

    chain = prompt | llm

    response = chain.invoke({
        "resume": state["resume"],
        "role": state["role"]
    })

    return {
        "project_suggestions": content_to_text(response.content)
    }


# ============================================================
# NODE 4 - GITHUB ANALYSIS
# ============================================================

def github_check(state: CareerState):

    prompt = ChatPromptTemplate.from_template("""
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
""")

    chain = prompt | llm

    response = chain.invoke({
        "github_id": state["github_id"],
        "resume": state["resume"]
    })

    return {
        "github_analysis": content_to_text(response.content)
    }


# ============================================================
# NODE 5 - FINAL SYNTHESIS
# ============================================================

def final_synthesis(state: CareerState):

    prompt = ChatPromptTemplate.from_template("""
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

Use clear headings and bullet points.

Do not mention internal nodes or LangGraph.
""")

    chain = prompt | llm

    response = chain.invoke({
        "job_results": state["job_results"],
        "skill_gap": state["skill_gap"],
        "project_suggestions": state["project_suggestions"],
        "github_analysis": state["github_analysis"],
        "question": state["question"]
    })

    return {
        "final_answer": content_to_text(response.content)
    }


# ============================================================
# LANGGRAPH
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
# LANGSERVE INPUT
# ============================================================

class CareerAgentInput(BaseModel):

    question: str = Field(
        description="The student's career question"
    )

    resume: str = Field(
        description="The student's resume text"
    )

    role: str = Field(
        description="The target job role"
    )

    github_id: str = Field(
        description="The student's GitHub username"
    )


# ============================================================
# LANGSERVE OUTPUT
# ============================================================

def extract_final_answer(graph_output: dict) -> str:

    if isinstance(graph_output, dict):

        answer = graph_output.get(
            "final_answer",
            ""
        )

        return content_to_text(answer)

    return content_to_text(graph_output)


formatted_graph_chain = (
    graph
    | RunnableLambda(extract_final_answer)
).with_types(
    input_type=CareerAgentInput,
    output_type=str
)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Placement Ready AI Career Agent",
    version="1.0"
)


# ============================================================
# LANGSERVE PLAYGROUND
# ============================================================

add_routes(
    app,
    formatted_graph_chain,
    path="/agent",
    playground_type="default"
)


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_pdf_text(pdf_bytes: bytes) -> str:

    try:

        pdf_file = io.BytesIO(pdf_bytes)

        reader = PdfReader(pdf_file)

        pages = []

        for page in reader.pages:

            text = page.extract_text()

            if text:
                pages.append(text)

        return "\n".join(pages).strip()

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=f"Could not read PDF: {str(e)}"
        )


# ============================================================
# RESUME ANALYSIS API
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
            detail="Please upload a resume PDF."
        )

    if not resume.filename.lower().endswith(".pdf"):

        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    pdf_bytes = await resume.read()

    if not pdf_bytes:

        raise HTTPException(
            status_code=400,
            detail="Uploaded PDF is empty."
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

    final_answer = content_to_text(
        result.get(
            "final_answer",
            "No final answer generated."
        )
    )

    return {

        "success": True,

        "filename": resume.filename,

        "extracted_characters": len(resume_text),

        "final_answer": final_answer
    }


# ============================================================
# FRONTEND
# ============================================================

@app.get("/", response_class=HTMLResponse)
def home():

    return """
<!DOCTYPE html>

<html>

<head>

    <title>Placement Ready AI</title>

    <meta name="viewport"
          content="width=device-width, initial-scale=1">

    <style>

        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            font-family: Arial, sans-serif;
            background: #f5f7fb;
            color: #111827;
        }

        .container {
            max-width: 1050px;
            margin: 40px auto;
            padding: 20px;
        }

        .header {
            text-align: center;
            margin-bottom: 30px;
        }

        .header h1 {
            font-size: 38px;
            margin-bottom: 10px;
        }

        .header p {
            color: #6b7280;
            font-size: 17px;
        }

        .card {
            background: white;
            border-radius: 18px;
            padding: 30px;
            box-shadow: 0 8px 30px rgba(0,0,0,0.08);
        }

        label {
            display: block;
            font-weight: bold;
            margin-bottom: 8px;
            margin-top: 20px;
        }

        input,
        textarea {
            width: 100%;
            padding: 14px;
            border: 1px solid #d1d5db;
            border-radius: 10px;
            font-size: 15px;
            outline: none;
        }

        input:focus,
        textarea:focus {
            border-color: #4f46e5;
        }

        input[type="file"] {
            background: #f9fafb;
        }

        textarea {
            min-height: 120px;
            resize: vertical;
        }

        button {
            width: 100%;
            margin-top: 25px;
            padding: 16px;
            border: none;
            border-radius: 12px;
            background: #4f46e5;
            color: white;
            font-size: 18px;
            font-weight: bold;
            cursor: pointer;
        }

        button:hover {
            background: #4338ca;
        }

        button:disabled {
            background: #9ca3af;
            cursor: not-allowed;
        }

        .loading {
            display: none;
            margin-top: 20px;
            text-align: center;
            color: #4f46e5;
            font-weight: bold;
        }

        .error {
            display: none;
            margin-top: 20px;
            padding: 15px;
            border-radius: 10px;
            background: #fee2e2;
            color: #991b1b;
        }

        .success {
            display: none;
            margin-top: 30px;
        }

        .success h2 {
            font-size: 28px;
            margin-bottom: 15px;
        }

        .info {
            padding: 15px;
            border: 1px solid #a7f3d0;
            background: #ecfdf5;
            color: #065f46;
            border-radius: 10px;
            margin-bottom: 20px;
        }

        .report {
            white-space: pre-wrap;
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            padding: 25px;
            border-radius: 14px;
            line-height: 1.7;
            font-size: 16px;
        }

    </style>

</head>


<body>

<div class="container">

    <div class="header">

        <h1>🤖 Placement Ready AI</h1>

        <p>
            Upload your resume and get an AI-powered
            career readiness report.
        </p>

    </div>


    <div class="card">

        <form id="resumeForm">

            <label>Resume PDF</label>

            <input
                type="file"
                id="resume"
                name="resume"
                accept=".pdf"
                required
            >


            <label>Target Job Role</label>

            <input
                type="text"
                id="role"
                name="role"
                placeholder="Example: AI Engineer"
                required
            >


            <label>GitHub Username</label>

            <input
                type="text"
                id="github_id"
                name="github_id"
                placeholder="Example: Bharath-max143"
                required
            >


            <label>Your Question</label>

            <textarea
                id="question"
                name="question"
                placeholder="Example: What should I learn to become placement ready?"
                required
            ></textarea>


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
            ⏳ Analyzing resume... Please wait.
        </div>


        <div
            class="error"
            id="error"
        ></div>


        <div
            class="success"
            id="success"
        >

            <h2>
                📊 Career Readiness Report
            </h2>

            <div
                class="info"
                id="info"
            ></div>

            <div
                class="report"
                id="report"
            ></div>

        </div>

    </div>

</div>


<script>

const form = document.getElementById("resumeForm");

const button = document.getElementById("analyzeBtn");

const loading = document.getElementById("loading");

const errorBox = document.getElementById("error");

const successBox = document.getElementById("success");

const infoBox = document.getElementById("info");

const reportBox = document.getElementById("report");


form.addEventListener("submit", async function(event) {

    event.preventDefault();


    errorBox.style.display = "none";

    successBox.style.display = "none";

    loading.style.display = "block";

    button.disabled = true;

    button.innerText = "⏳ Analyzing...";


    const formData = new FormData();

    formData.append(
        "resume",
        document.getElementById("resume").files[0]
    );

    formData.append(
        "role",
        document.getElementById("role").value
    );

    formData.append(
        "github_id",
        document.getElementById("github_id").value
    );

    formData.append(
        "question",
        document.getElementById("question").value
    );


    try {

        const response = await fetch(
            "/analyze",
            {
                method: "POST",
                body: formData
            }
        );


        const data = await response.json();


        if (!response.ok) {

            throw new Error(
                data.detail || "Something went wrong."
            );

        }


        infoBox.innerText =
            "Resume analyzed successfully. " +
            "Extracted " +
            data.extracted_characters +
            " characters from " +
            data.filename;


        reportBox.innerText =
            data.final_answer;


        successBox.style.display = "block";


    } catch (error) {

        errorBox.innerText =
            "❌ " + error.message;

        errorBox.style.display = "block";

    }


    loading.style.display = "none";

    button.disabled = false;

    button.innerText = "🚀 Analyze Resume";

});

</script>


</body>

</html>
"""


# ============================================================
# RUN SERVER
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
