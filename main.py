import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.ollama import OllamaChatCompletionClient


async def main():
    model_client = OllamaChatCompletionClient(
        model="llama3.2"
    )

    job_analyst = AssistantAgent(
        name="JobAnalyst",
        model_client=model_client,
        system_message=(
            "You are a job analyst. "
            "Analyze job descriptions and identify the most important skills, "
            "requirements, responsibilities, and seniority level. "
            "Keep your answer concise."
        ),
    )

    cv_analyst = AssistantAgent(
        name="CVAnalyst",
        model_client=model_client,
        system_message=(
            "You are a CV analyst. "
            "Compare a candidate profile with job requirements. "
            "Identify strong matches and gaps. "
            "Keep your answer concise."
        ),
    )

    career_reviewer = AssistantAgent(
        name="CareerReviewer",
        model_client=model_client,
        system_message=(
            "You are a career reviewer. "
            "Use the job analysis and CV analysis to make a final hiring-fit assessment. "
            "Return a fit score from 0 to 100, APPLY or DON'T APPLY, "
            "top 3 strengths, top 3 gaps, and a short final recommendation."
        ),
    )

    job_description = """
    We are looking for an AI Automation Specialist.

    Required skills:
    - Experience with AI agents
    - Agent orchestration frameworks such as LangGraph, CrewAI or AutoGen
    - Function calling
    - RAG
    - Memory systems for agents
    - Ability to understand business processes
    """

    candidate_profile = """
    Candidate profile:
    - 10+ years of operations and process management experience
    - Experience managing KPIs and business procedures
    - Built AI automation projects using n8n, Supabase and Gemini
    - Built InvoiceAI for invoice processing automation
    - Built an AI Contract Analyzer
    - Google Cloud Generative AI Leader certification
    - Google AI Essentials certification
    - Limited direct experience with LangGraph, CrewAI, AutoGen and RAG
    """

    job_result = await job_analyst.run(
        task=job_description
    )

    job_analysis = job_result.messages[-1].content

    print("\n--- JOB ANALYSIS ---\n")
    print(job_analysis)

    cv_task = f"""
    JOB ANALYSIS:
    {job_analysis}

    CANDIDATE PROFILE:
    {candidate_profile}

    Compare the candidate with the job requirements.

    Return:
    1. Strong matches
    2. Gaps
    """

    cv_result = await cv_analyst.run(
        task=cv_task
    )

    cv_analysis = cv_result.messages[-1].content

    print("\n--- CV ANALYSIS ---\n")
    print(cv_analysis)

    review_task = f"""
    JOB ANALYSIS:
    {job_analysis}

    CV ANALYSIS:
    {cv_analysis}

    Make the final assessment.

    Return exactly:
    FIT SCORE:
    DECISION:
    TOP 3 STRENGTHS:
    TOP 3 GAPS:
    FINAL RECOMMENDATION:
    """

    review_result = await career_reviewer.run(
        task=review_task
    )

    print("\n--- FINAL REVIEW ---\n")
    print(review_result.messages[-1].content)

    await model_client.close()


asyncio.run(main())