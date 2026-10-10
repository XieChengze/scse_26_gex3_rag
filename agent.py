"""
University IT support agent.

A LangChain agent backed by a single RAG tool (search_university_knowledge),
followed by a reflection loop:

    draft  ->  strict fact-checking review  ->  revision (if needed)

The reviewer is deliberately strict about contact details (phone numbers,
emails, URLs): any such detail that does not appear VERBATIM in the retrieved
policy text forces a revision, and when the knowledge base simply does not
contain the requested detail the final answer must plainly say so instead of
hallucinating it.
"""

import re
from typing import Literal

from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import ToolMessage

from rag_tool import search_university_knowledge



MODEL_NAME = "qwen3:8b"

MAX_REVISIONS = 2


model = ChatOllama(model=MODEL_NAME, temperature=0)

if "think" in model.model_fields:
    model.think = False


SYSTEM_PROMPT = """
You are the University IT Service Desk support assistant.

You answer ONLY from information returned by the
search_university_knowledge tool, which searches the official University IT
policy knowledge base.

Rules you must always follow:
1. Always call search_university_knowledge before answering a policy question.
2. Use only facts that appear in the retrieved policy text. Do not use your
   own memory of universities, phone numbers or websites.
3. Never invent, guess or "complete" any phone number, email address, URL,
   office location or other contact detail. If the exact requested detail
   (for example an exact phone number) is not present in the retrieved text,
   you must clearly say that you do not know and that the detail is not
   listed in the knowledge base. You may still give the guidance that IS
   documented (for example that the user should contact the IT Service Desk).
4. When the requested detail is missing, do NOT provide any alternative
   instructions or redirect the user anywhere else - do not suggest a
   website, portal, directory, email, another phone line or asking someone
   else. Simply state that the information is not available in the knowledge
   base, together with documented guidance only.
5. Do not give unrelated information.
6. Keep the answer clear, concise and helpful.
"""


agent = create_agent(
    model=model,
    tools=[search_university_knowledge],
    system_prompt=SYSTEM_PROMPT,
)



# Reflection: structured verdict from a strict reviewer.

class Reflection(BaseModel):
    verdict: Literal["PASS", "REVISE"]
    problem: str
    suggestion: str


reflection_model = model.with_structured_output(Reflection)


review_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a strict fact-checker reviewing an answer produced by a
            University IT support agent.

            You are given the user's question, the exact policy text the agent
            RETRIEVED from its knowledge base, and the agent's draft answer.

            Return REVISE if any of the following is true:
            1. The answer states a phone number, email address, URL or other
               contact detail that does NOT appear verbatim in the retrieved
               policy text - even if it looks plausible or familiar.
            2. The answer guesses, hedges or fills gaps from common knowledge
               when the retrieved text does not contain the information.
            3. The retrieved text does not contain the requested detail (for
               example an exact phone number) but the answer fails to plainly
               state that the information is not available / not known.
            4. When the requested detail is missing, the answer redirects the
               user to another channel to obtain it (a website, portal,
               directory, email, another phone line, or asking someone else),
               or gives any other instruction beyond the documented guidance.
            5. The answer contradicts the retrieved text or does not actually
               answer the question, or contains irrelevant material.

            When the requested detail is absent from the retrieved text, the
            only acceptable answer says, in substance, that the assistant does
            not know the number / it is not listed in the knowledge base. It
            may additionally state documented guidance such as "contact the IT
            Service Desk", but it must not point anywhere else. Return PASS
            only for answers of that form, or for answers fully supported by
            the retrieved text.

            If you return REVISE, describe the precise problem and the exact
            correction required.
            """,
        ),
        (
            "human",
            """
            USER QUESTION:

            {question}


            RETRIEVED POLICY TEXT (the only evidence available):

            {context}


            AGENT ANSWER:

            {answer}
            """,
        ),
    ]
)

review_chain = review_prompt | reflection_model


revision_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Revise the University IT support answer using the reviewer's
            feedback and ONLY the retrieved policy text.

            Remove every statement or detail that is not supported by the
            retrieved text, in particular every phone number, email address
            or URL that does not appear verbatim in it.

            If the requested information is absent from the retrieved text,
            plainly say that you do not know / that it is not listed in the
            knowledge base, and then give only guidance that IS documented
            (for example that the user should contact the IT Service Desk).
            Remove every redirect to a website, portal, directory, email or
            other channel.

            Keep the answer clear and concise.
            """,
        ),
        (
            "human",
            """
            USER QUESTION:

            {question}


            RETRIEVED POLICY TEXT (the only evidence available):

            {context}


            ORIGINAL ANSWER:

            {answer}


            REVIEW PROBLEM:

            {problem}


            SUGGESTED IMPROVEMENT:

            {suggestion}


            Produce the revised answer.
            """,
        ),
    ]
)

revision_chain = revision_prompt | model


_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def _clean(text: str) -> str:
    """Remove any leftover Qwen reasoning block from displayed text."""
    return _THINK_BLOCK.sub("", text or "").strip()


def answer_with_reflection(question: str, verbose: bool = True) -> str:
    """Draft an answer with the RAG agent, then review and revise it."""

    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]}
    )

    messages = result["messages"]

    # The policy text actually retrieved through the RAG tool is the only
    # evidence the reviewer (and reviser) may trust.
    tool_outputs = [
        message.content
        for message in messages
        if isinstance(message, ToolMessage)
    ]
    context = (
        "\n\n".join(tool_outputs)
        if tool_outputs
        else "(No policy text was retrieved.)"
    )

    answer = _clean(messages[-1].content)

    if verbose:
        print("\nDRAFT ANSWER:")
        print(answer)

    for round_number in range(1, MAX_REVISIONS + 2):
        reflection = review_chain.invoke(
            {
                "question": question,
                "context": context,
                "answer": answer,
            }
        )

        if verbose:
            print(f"\nREFLECTION (round {round_number}): {reflection.verdict}")
            if reflection.verdict == "REVISE":
                print("Problem:   ", reflection.problem)
                print("Suggestion:", reflection.suggestion)

        if reflection.verdict == "PASS":
            break

        if round_number > MAX_REVISIONS:
            break

        revised = revision_chain.invoke(
            {
                "question": question,
                "context": context,
                "answer": answer,
                "problem": reflection.problem,
                "suggestion": reflection.suggestion,
            }
        )
        answer = _clean(revised.content)

        if verbose:
            print("\nREVISED ANSWER:")
            print(answer)

    return answer
