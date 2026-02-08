from src.state.graph_State import GraphState
from langgraph.graph import StateGraph, START, END
from src.tools.PDF_tool import build_pdf_retriver
from src.Nodes.chat_with_pdf import init_components
from src.Nodes.chat_with_pdf import (
    retrieve,
    grade_docs,
    transform_query,
    generate,
    decide_to_generate,
    grade_generation_v_documents_and_question,
)
class Graph_builder:
    def __init__(self,pdf_path):
        self.pdf_path = pdf_path
    def build(self):
        try:
            retriver = build_pdf_retriver(self.pdf_path)
        except Exception as exc:
            raise RuntimeError(f"build_pdf_retriver failed: {exc}") from exc

        self.graph = StateGraph(GraphState)

        try:
            (
                retrieval_grader,
                rag_chain,
                hallucination_grader,
                answer_grader,
                question_rewriter,
            ) = init_components()
        except Exception as exc:
            raise RuntimeError(f"init_components failed: {exc!r}") from exc

        self.graph.add_node("retrive", lambda s: retrieve(s, retriver))
        self.graph.add_node("grade_docs", lambda s: grade_docs(s, retrieval_grader))
        self.graph.add_node("transform_query", lambda s: transform_query(s, question_rewriter))
        self.graph.add_node("generate", lambda s: generate(s, rag_chain))

        self.graph.add_edge(START, "retrive")
        self.graph.add_edge("retrive", "grade_docs")
        self.graph.add_conditional_edges(
            "grade_docs",
             decide_to_generate,{
                 "transform_query":"transform_query",
                 "generate":"generate",
             },
        )
        self.graph.add_edge("transform_query", "retrive")
        # self.graph.add_conditional_edges(
        #     "generate",
        #     lambda s : grade_generation_v_documents_and_question(s, hallucination_grader, answer_grader),
        #          {
        #             "not supported": "generate",
        #             "useful": END,
        #             "not useful": "transform_query",
        #             "stop": END,
        #         },
        # )

        self.graph.add_edge("generate", END)


        try:
            return self.graph.compile()
        except Exception as exc:
            raise RuntimeError(f"graph.compile failed: {exc!r}") from exc

    # Add methods for graph building as needed