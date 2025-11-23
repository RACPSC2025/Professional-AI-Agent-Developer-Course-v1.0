"""
Module 5 Enhancement: Self-RAG Implementation (2024)
Framework: LangChain + LangGraph
Objective: Self-Reflective Retrieval-Augmented Generation

This example demonstrates:
- Self-RAG with reflection tokens
- Iterative refinement of retrieval and generation
- Dynamic decision-making for when to retrieve
- Quality assessment and self-correction
"""

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import TypedDict, Annotated, List
import operator
import os

# ============================================================================
# 1. STATE DEFINITION
# ============================================================================

class SelfRAGState(TypedDict):
    """State for Self-RAG workflow."""
    question: str
    retrieved_docs: Annotated[List[str], operator.add]
    generation: str
    retrieval_decision: str  # "yes" or "no"
    relevance_score: float  # 0.0 to 1.0
    support_score: float  # 0.0 to 1.0
    utility_score: float  # 0.0 to 1.0
    iterations: int
    final_answer: str


# ============================================================================
# 2. SETUP COMPONENTS
# ============================================================================

def setup_vectorstore():
    """Create a simple vectorstore with sample documents."""
    documents = [
        "LangGraph is a library for building stateful, multi-actor applications with LLMs. It extends LangChain with graph-based state management.",
        "Self-RAG enables language models to adaptively retrieve passages on-demand and generate responses while reflecting on their own generation process.",
        "CrewAI Flows provide an event-driven architecture for orchestrating AI workflows with @start and @listen decorators.",
        "AutoGen v0.4 introduces an asynchronous, event-driven architecture based on the actor model for multi-agent systems.",
        "Corrective RAG (CRAG) improves retrieval quality by evaluating and correcting retrieved documents before generation.",
        "Vector databases like ChromaDB, Pinecone, and Weaviate enable semantic search at scale using embeddings.",
        "Hybrid search combines vector similarity search with keyword-based BM25 for improved retrieval accuracy.",
        "Pydantic models provide type-safe structured outputs from LLMs with automatic validation."
    ]
    
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_texts(documents, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 3})


# ============================================================================
# 3. SELF-RAG NODES
# ============================================================================

def decide_to_retrieve(state: SelfRAGState) -> SelfRAGState:
    """
    Reflection Token 1: Decide if retrieval is needed.
    Uses LLM to determine if external knowledge is required.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    prompt = ChatPromptTemplate.from_template("""
    Question: {question}
    
    Do you need to retrieve external information to answer this question accurately?
    Consider:
    - Is this a factual question requiring specific knowledge?
    - Can you answer confidently with your training data?
    
    Respond with ONLY "yes" or "no".
    """)
    
    chain = prompt | llm | StrOutputParser()
    decision = chain.invoke({"question": state["question"]}).strip().lower()
    
    state["retrieval_decision"] = decision
    print(f"🤔 Retrieval Decision: {decision}")
    
    return state


def retrieve_documents(state: SelfRAGState) -> SelfRAGState:
    """Retrieve relevant documents if needed."""
    if state["retrieval_decision"] == "no":
        print("⏭️  Skipping retrieval (not needed)")
        return state
    
    retriever = setup_vectorstore()
    docs = retriever.get_relevant_documents(state["question"])
    
    retrieved_texts = [doc.page_content for doc in docs]
    state["retrieved_docs"] = retrieved_texts
    
    print(f"📚 Retrieved {len(retrieved_texts)} documents")
    return state


def assess_relevance(state: SelfRAGState) -> SelfRAGState:
    """
    Reflection Token 2: Assess relevance of retrieved documents.
    Scores how relevant the retrieved docs are to the question.
    """
    if not state["retrieved_docs"]:
        state["relevance_score"] = 1.0  # No retrieval needed
        return state
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    prompt = ChatPromptTemplate.from_template("""
    Question: {question}
    
    Retrieved Documents:
    {documents}
    
    On a scale of 0.0 to 1.0, how relevant are these documents to answering the question?
    - 1.0 = Highly relevant, directly addresses the question
    - 0.5 = Partially relevant, some useful information
    - 0.0 = Not relevant, doesn't help answer the question
    
    Respond with ONLY a number between 0.0 and 1.0
    """)
    
    docs_text = "\n\n".join(state["retrieved_docs"])
    chain = prompt | llm | StrOutputParser()
    
    try:
        score = float(chain.invoke({
            "question": state["question"],
            "documents": docs_text
        }).strip())
        state["relevance_score"] = max(0.0, min(1.0, score))
    except:
        state["relevance_score"] = 0.5  # Default if parsing fails
    
    print(f"📊 Relevance Score: {state['relevance_score']:.2f}")
    return state


def generate_response(state: SelfRAGState) -> SelfRAGState:
    """Generate response using retrieved documents (if any)."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    if state["retrieved_docs"] and state["relevance_score"] > 0.3:
        # Use retrieved context
        prompt = ChatPromptTemplate.from_template("""
        Answer the question based on the following context:
        
        Context:
        {context}
        
        Question: {question}
        
        Provide a clear, accurate answer based on the context.
        """)
        
        context = "\n\n".join(state["retrieved_docs"])
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({
            "question": state["question"],
            "context": context
        })
    else:
        # Generate without context
        prompt = ChatPromptTemplate.from_template("""
        Question: {question}
        
        Provide a clear, accurate answer based on your knowledge.
        """)
        
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({"question": state["question"]})
    
    state["generation"] = response
    print(f"✍️  Generated response ({len(response)} chars)")
    return state


def assess_support(state: SelfRAGState) -> SelfRAGState:
    """
    Reflection Token 3: Assess if generation is supported by retrieved docs.
    Checks for hallucinations and factual grounding.
    """
    if not state["retrieved_docs"]:
        state["support_score"] = 1.0  # No docs to check against
        return state
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    prompt = ChatPromptTemplate.from_template("""
    Retrieved Documents:
    {documents}
    
    Generated Answer:
    {answer}
    
    On a scale of 0.0 to 1.0, how well is the answer supported by the documents?
    - 1.0 = Fully supported, all claims are backed by the documents
    - 0.5 = Partially supported, some claims lack evidence
    - 0.0 = Not supported, contains information not in documents
    
    Respond with ONLY a number between 0.0 and 1.0
    """)
    
    docs_text = "\n\n".join(state["retrieved_docs"])
    chain = prompt | llm | StrOutputParser()
    
    try:
        score = float(chain.invoke({
            "documents": docs_text,
            "answer": state["generation"]
        }).strip())
        state["support_score"] = max(0.0, min(1.0, score))
    except:
        state["support_score"] = 0.5
    
    print(f"🎯 Support Score: {state['support_score']:.2f}")
    return state


def assess_utility(state: SelfRAGState) -> SelfRAGState:
    """
    Reflection Token 4: Assess overall utility of the response.
    Evaluates if the answer is actually useful for the user.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    prompt = ChatPromptTemplate.from_template("""
    Question: {question}
    
    Generated Answer:
    {answer}
    
    On a scale of 0.0 to 1.0, how useful is this answer?
    - 1.0 = Highly useful, directly answers the question clearly
    - 0.5 = Somewhat useful, provides partial information
    - 0.0 = Not useful, doesn't address the question
    
    Respond with ONLY a number between 0.0 and 1.0
    """)
    
    chain = prompt | llm | StrOutputParser()
    
    try:
        score = float(chain.invoke({
            "question": state["question"],
            "answer": state["generation"]
        }).strip())
        state["utility_score"] = max(0.0, min(1.0, score))
    except:
        state["utility_score"] = 0.5
    
    print(f"⭐ Utility Score: {state['utility_score']:.2f}")
    return state


def should_refine(state: SelfRAGState) -> str:
    """
    Decision node: Should we refine the answer?
    Refines if scores are low and we haven't iterated too much.
    """
    state["iterations"] = state.get("iterations", 0) + 1
    
    # Refine if any score is low and we haven't exceeded max iterations
    needs_refinement = (
        state["relevance_score"] < 0.7 or
        state["support_score"] < 0.7 or
        state["utility_score"] < 0.7
    )
    
    if needs_refinement and state["iterations"] < 2:
        print(f"🔄 Refining (iteration {state['iterations']})...")
        return "refine"
    else:
        state["final_answer"] = state["generation"]
        print(f"✅ Finalizing answer")
        return "finalize"


# ============================================================================
# 4. BUILD GRAPH
# ============================================================================

def build_self_rag_graph():
    """Build the Self-RAG workflow graph."""
    workflow = StateGraph(SelfRAGState)
    
    # Add nodes
    workflow.add_node("decide_retrieve", decide_to_retrieve)
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("assess_relevance", assess_relevance)
    workflow.add_node("generate", generate_response)
    workflow.add_node("assess_support", assess_support)
    workflow.add_node("assess_utility", assess_utility)
    
    # Add edges
    workflow.set_entry_point("decide_retrieve")
    workflow.add_edge("decide_retrieve", "retrieve")
    workflow.add_edge("retrieve", "assess_relevance")
    workflow.add_edge("assess_relevance", "generate")
    workflow.add_edge("generate", "assess_support")
    workflow.add_edge("assess_support", "assess_utility")
    
    # Conditional edge for refinement
    workflow.add_conditional_edges(
        "assess_utility",
        should_refine,
        {
            "refine": "retrieve",  # Loop back to retrieve
            "finalize": END
        }
    )
    
    return workflow.compile()


# ============================================================================
# 5. MAIN EXECUTION
# ============================================================================

def main():
    """Run Self-RAG examples."""
    print("\n🧠 Self-RAG: Self-Reflective Retrieval-Augmented Generation")
    print("=" * 70)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not set")
        return
    
    # Build the graph
    app = build_self_rag_graph()
    
    # Example questions
    questions = [
        "What is LangGraph and how does it work?",
        "Explain Self-RAG and its benefits",
        "What are the key features of CrewAI Flows?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'='*70}")
        print(f"EXAMPLE {i}: {question}")
        print('='*70)
        
        # Run the graph
        result = app.invoke({
            "question": question,
            "retrieved_docs": [],
            "generation": "",
            "retrieval_decision": "",
            "relevance_score": 0.0,
            "support_score": 0.0,
            "utility_score": 0.0,
            "iterations": 0,
            "final_answer": ""
        })
        
        # Display results
        print(f"\n📝 FINAL ANSWER:")
        print(f"{result['final_answer']}\n")
        print(f"📊 Quality Metrics:")
        print(f"   Relevance: {result['relevance_score']:.2f}")
        print(f"   Support: {result['support_score']:.2f}")
        print(f"   Utility: {result['utility_score']:.2f}")
        print(f"   Iterations: {result['iterations']}")
    
    print("\n✅ All examples completed!")
    print("\n💡 Self-RAG Features Demonstrated:")
    print("   - Dynamic retrieval decisions")
    print("   - Relevance assessment")
    print("   - Support verification (anti-hallucination)")
    print("   - Utility evaluation")
    print("   - Iterative refinement")


if __name__ == "__main__":
    main()
