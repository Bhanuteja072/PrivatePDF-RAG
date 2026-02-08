from src.state.graph_State import GraphState
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


def init_components():
    llm = OllamaLLM(model="llama3.2:1b", temperature=0)
    system = """You are a lenient relevance grader. If the document contains any keyword(s) or semantic meaning related to the user question, grade it as relevant. The goal is to filter out only clearly wrong hits.
Answer with exactly 'yes' or 'no'. If unsure, prefer 'yes'."""

    grade_prompt=ChatPromptTemplate.from_messages(
        [
            ("system",system),
            ("human","Retrieved document: \n\n {document} \n\n User question: {question}")
        ]
    )
    retrieval_grader = grade_prompt | llm | StrOutputParser()

    # Simple RAG prompt without langchain_hub dependency
    prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant. Use only the provided context to answer.

Context:
{context}

Question:
{question}

Instructions:
- If the context explains the answer in paragraph form, respond in paragraph form.
- If the context presents the answer as multiple distinct points, you may list them naturally.
- Do not force bullet points if they are not present in the context.
- Preserve the original meaning and structure of the context as much as possible.
- Do not add, infer, or assume any information outside the context.
- If the context does not contain the answer, clearly state that.

Formatting:
- Keep the response clean and readable.
- Do not split a single idea into artificial points.
"""
    )


#     Instructions:
# - Answer using clearly separated bullet points.
# - Start each bullet with a **Bold topic name**.
# - Write 2 sentences per bullet explaining the concept using the context.
# - Insert a blank line between every bullet.
# - Define concepts first, then explain purpose or differences if relevant.
# - Do NOT infer, assume, or add external knowledge.

# Formatting:
# - Use standard bullet points.
# - No paragraphs or merged bullets
# """

    rag_chain = prompt | llm | StrOutputParser()

    # Prompt
    system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.
Answer with exactly 'yes' or 'no'. If the generation is even partially grounded, prefer 'yes'."""


    hallicinnation_prompt=ChatPromptTemplate.from_messages(
        [
            ("system",system),
            ("human","Retrieved facts: \n\n {facts} \n\n LLM generation: {generation}")
        ]
    )


    hallucination_grader = hallicinnation_prompt | llm | StrOutputParser()


    # Prompt
    system = """You are a grader assessing whether an answer addresses / resolves a question.
Answer with exactly 'yes' or 'no'. If the answer partially addresses the question, prefer 'yes'."""
    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
        ]
    )

    answer_grader = answer_prompt | llm | StrOutputParser()

    system = """You a question re-writer that converts an input question to a better version that is optimized \n
     for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Here is the initial question: \n\n {question} \n Formulate an improved and accurate question in a single line.",
            ),
        ]
    )


    question_rewriter = re_write_prompt | llm | StrOutputParser()
    return (retrieval_grader, rag_chain, hallucination_grader, answer_grader, question_rewriter)


def retrieve(state : GraphState,retriever):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
     """
    print("---RETRIEVE---")
    question=state["question"]
    if retriever is None:
        raise ValueError("Retriever is not initialized. Please re-upload the PDF.")

    docs=retriever.invoke(question)

    failures = state.get("failures", 0)

    return {"documents":docs , "question":question, "failures": failures}


def generate(state : GraphState ,rag_chain):
  """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
  """
  print("---GENERATE---")
  question=state["question"]
  docs=state["documents"]
  generation=rag_chain.invoke({"context":docs,"question":question})
  failures = state.get("failures", 0)
  return {"documents": docs, "question": question, "generation": generation, "failures": failures}



def _is_yes(text: str) -> bool:
    return str(text).strip().lower().startswith("yes")


def grade_docs(state : GraphState ,retrieval_grader):
    """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with only filtered relevant documents
    """
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question=state["question"]
    docs=state["documents"]
    failures = state.get("failures", 0)
    # Score each doc
    filtered_docs = []
    for d in docs:
        score = retrieval_grader.invoke(
                {"question": question, "document": d.page_content}
            )
        if _is_yes(score):
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue
    return {"documents": filtered_docs, "question": question, "failures": failures}

def transform_query(state : GraphState ,question_rewriter):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
  """
    question=state["question"]
    documents = state["documents"]
    failures = state.get("failures", 0)

    print("---TRANSFORM QUERY---")
    new_question=question_rewriter.invoke({"question":question})
    return {"documents": documents, "question": new_question, "failures": failures}



def decide_to_generate(state : GraphState):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    state["question"]
    filtered_documents = state["documents"]

    if not filtered_documents:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
        )
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"
    








def grade_generation_v_documents_and_question(state : GraphState ,hallucination_grader, answer_grader):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"facts": documents, "generation": generation}
    )
    grade = _is_yes(score)
    failures = state.get("failures", 0)

    # Check hallucination
    if grade:
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = _is_yes(score)
        if grade:
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            state["failures"] = failures
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            failures += 1
            state["failures"] = failures
            if failures >= 2:
                return "stop"
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        failures += 1
        state["failures"] = failures
        if failures >= 2:
            return "stop"
        return "not supported"
