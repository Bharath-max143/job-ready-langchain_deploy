import os
import uvicorn
from typing import TypedDict

from fastapi import FastAPI
from langserve import add_routes

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, END


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
# 4. NODES (Job Search, Skill Gap, Projects, GitHub, Synthesis)
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
    response = chain.invoke({"resume": state["resume"], "role": state["role"]})
    return {"job_results": response.content}


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
    response = chain.invoke({"resume": state["resume"], "role": state["role"]})
    return {"skill_gap": response.content}


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
    response = chain.invoke({"resume": state["resume"], "role": state["role"]})
    return {"project_suggestions": response.content}


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
    response = chain.invoke({"github_id": state["github_id"], "resume": state["resume"]})
    return {"github_analysis": response.content}


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
    return {"final_answer": response.content}


# ============================================================
# 5. BUILD LANGGRAPH
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
# 6. INPUT/OUTPUT SCHEMAS & FORMATTERS FOR LANGSERVE
# ============================================================

class CareerAgentInput(BaseModel):
    question: str = Field(description="The student's career question")
    resume: str = Field(description="The student's resume text")
    role: str = Field(description="The target job role")
    github_id: str = Field(description="The student's GitHub username")


def extract_final_answer(graph_output: dict) -> str:
    if isinstance(graph_output, dict) and "final_answer" in graph_output:
        return graph_output["final_answer"]
    return str(graph_output)


# Create a runnable chain that accepts the Pydantic input, runs the graph, and extracts the string
formatted_graph_chain = (
    graph
    | RunnableLambda(extract_final_answer)
).with_types(input_type=CareerAgentInput, output_type=str)


# ============================================================
# 7. FASTAPI APP & ROUTES
# ============================================================

app = FastAPI(
    title="Placement Ready AI Career Agent",
    version="1.0"
)

add_routes(
    app,
    formatted_graph_chain,
    path="/agent",
    playground_type="default"
)


@app.get("/")
def home():
    return {
        "message": "Placement Ready AI Career Agent is running!"
    }


# ============================================================
# 8. LOCAL RUN
# ============================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
