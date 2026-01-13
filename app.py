###
import os
from typing import List
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph

# --- Veri Hazırlığı ---
loader = TextLoader("bilgi.txt")
documents = loader.load()

# Embedding ve Vektör Veritabanı
print("Embedding modeli yükleniyor ve vektör veritabanı oluşturuluyor...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_db = Chroma.from_documents(documents=documents, embedding=embeddings)
retriever = vector_db.as_retriever()
print("Tamamlandı.")

# --- LLM Hazırlığı ---
llm = OllamaLLM(model="llama3.1", temperature=0)

# --- Prompts ve Zincirler ---

# 1. Retrieval Grader (Alaka Düzeyi Kontrolü)
prompt_relevance = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keywords related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    The documents and question might be in Turkish. Analyze semantically. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no premable or explaination.
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here is the retrieved document: \n\n {document} \n\n
    Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["question", "document"],
)
retrieval_grader = prompt_relevance | llm | JsonOutputParser()

# 2. Generator (Yanıt Üretici)
prompt_generate = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. \n
    If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise. \n
    The context and question are in Turkish. Answer in Turkish. \n
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Question: {question} \n
    Context: {context} \n
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["question", "context"],
)
rag_chain = prompt_generate | llm | StrOutputParser()

# 3. Hallucination Grader (Halüsinasyon Kontrolü)
prompt_hallucination = PromptTemplate(
    template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a grader assessing whether an answer is grounded in / supported by a set of facts. \n
    Give a binary score 'yes' or 'no' score to indicate whether the answer is grounded in / supported by a set of facts. \n
    The text might be in Turkish. \n
    Provide the binary score as a JSON with a single key 'score' and no premable or explaination.
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here are the facts:
    \n {documents} \n
    Here is the answer: {generation} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["generation", "documents"],
)
hallucination_grader = prompt_hallucination | llm | JsonOutputParser()

# 4. Answer Grader (Yanıtın Soruyla Alakası)
prompt_answer = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a grader assessing whether an answer addresses / resolves a question \n
    Give a binary score 'yes' or 'no' score to indicate whether the answer addresses the question. \n
    The question and answer are likely in Turkish. Focus on semantic resolution. \n
    Provide the binary score as a JSON with a single key 'score' and no premable or explaination.
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    User question: \n {question} \n
    LLM generation: {generation} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["generation", "question"],
)
answer_grader = prompt_answer | llm | JsonOutputParser()

# --- Graph State ---
from typing_extensions import TypedDict

class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]

# --- Nodes ---

def retrieve(state):
    """
    Retrieve documents from vectorstore
    """
    print("---RETRIEVE---")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question
    """
    print("---CHECK DOCUMENT RELEVANCE---")
    question = state["question"]
    documents = state["documents"]
    
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        grade = score['score']
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue
            
    return {"documents": filtered_docs, "question": question}

def generate(state):
    """
    Generate answer using RAG on retrieved documents
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    
    if not documents:
        return {"documents": [], "question": question, "generation": "Bu konuda bilgi bulamadım."}

    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

def check_hallucinations(state):
    """
    Check if generation is grounded in documents and answers question
    """
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    
    if generation == "Bu konuda bilgi bulamadım.":
        return "useful" # Bypassing checks if no info

    score = hallucination_grader.invoke({"documents": documents, "generation": generation})
    grade = score['score']
    
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Answer addresses question check removed as it was yielding false negatives.
        return "useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS---")
        return "not supported"

# --- Conditional Edges ---

def decide_to_generate(state):
    """
    Determines whether to generate an answer or end
    """
    print("---ASSESS GRADED DOCUMENTS---")
    filtered_documents = state["documents"]
    
    if not filtered_documents:
        print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION---")
        return "no_context"
    else:
        print("---DECISION: GENERATE---")
        return "generate"

# --- Graph Construction ---

workflow = StateGraph(GraphState)

# Define nodes
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)

# Build graph
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "generate": "generate",
        "no_context": END,
    },
)

def check_hallucination_logic(state):
    result = check_hallucinations(state)
    if result == "useful":
        return END
    else:
        # If hallucinated or not useful, we could regenerate or search web, 
        # but user specifically asked to say "bu konuda bilgi bulamadım".
        # So we hijack the state to set the message.
        return "reset_to_no_info"

# We need a small node to reset the answer if it fails checks, because conditional edges can't modify state directly
def set_no_info(state):
    print("---SETTING FINAL ANSWER TO 'BILGI BULAMADIM'---")
    return {"generation": "Bu konuda bilgi bulamadım."}

workflow.add_node("set_no_info", set_no_info)
workflow.add_conditional_edges(
    "generate",
    check_hallucination_logic,
    {
        END: END,
        "reset_to_no_info": "set_no_info"
    }
)
workflow.add_edge("set_no_info", END)

# Compile
app = workflow.compile()

# --- Main Loop ---
if __name__ == "__main__":
    print("\n--- LangGraph RAG Hazır! Sorularınızı sorabilirsiniz (Çıkmak için 'q' veya 'exit' yazın) ---")
    while True:
        query = input("\nSoru: ")
        if query.lower() in ["q", "exit", "quit", "çıkış"]:
            print("Görüşürüz!")
            break
        if not query.strip():
            continue

        inputs = {"question": query}
        # invoke returns the final state
        result = app.invoke(inputs)
        
        print("\n--- YANIT ---")
        # Handle cases where we ended early due to no context
        if "generation" in result:
             print(result["generation"])
        else:
             print("Bu konuda bilgi bulamadım.")