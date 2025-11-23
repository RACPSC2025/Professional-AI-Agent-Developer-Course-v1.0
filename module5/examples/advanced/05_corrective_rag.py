"""
Module 5 Enhancement: Corrective RAG (CRAG) - November 2025
Framework: LangChain + LangGraph 1.0
Objective: Improve retrieval quality through correction and refinement

This example demonstrates:
- Document relevance assessment
- Web search fallback for low-quality retrieval
- Knowledge refinement (filtering irrelevant parts)
- Corrective actions based on confidence scores
- LangGraph 1.0 state machine implementation
"""

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import DuckDuckGoSearchRun
from typing import TypedDict, Annotated, List
import operator
import os

# ============================================================================
# 1. STATE DEFINITION
# ============================================================================

class CRAGState(TypedDict):
    """State for Corrective RAG workflow."""
    question: str
    retrieved_docs: Annotated[List[str], operator.add]
    relevance_scores: List[float]
    overall_confidence: float
    web_search_results: str
    refined_context: str
    generation: str
    correction_applied: bool


# ============================================================================
# 2. SETUP COMPONENTS
# ============================================================================

def setup_vectorstore():
    """Create vectorstore with sample documents."""
    documents = [
        "LangGraph 1.0 was released in October 2025 with durable execution and advanced streaming capabilities.",
        "OpenAI released GPT-5.1 on November 13, 2025, featuring enhanced steerability and faster responses.",
        "CrewAI v1.1.0 was released on October 21, 2025, with multi-provider LLM support and improved performance.",
        "Agentic RAG emerged as a major trend in November 2025, enabling iterative reasoning and dynamic retrieval.",
        "Multimodal RAG integrates visual and text information for improved context understanding.",
        "Graph-RAG uses knowledge graphs to enable multi-hop reasoning and deterministic traversal.",
        "The global RAG market is projected to reach $1.85 billion in 2025.",
        "Microsoft announced the Agent Framework at Ignite 2025, unifying Semantic Kernel and AutoGen."
    ]
    
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_texts(documents, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 3})


# ============================================================================
# 3. CRAG NODES
# ============================================================================

def retrieve_documents(state: CRAGState) -> CRAGState:
    """Initial retrieval from vector database."""
    retriever = setup_vectorstore()
    docs = retriever.get_relevant_documents(state["question"])
    
    retrieved_texts = [doc.page_content for doc in docs]
    state["retrieved_docs"] = retrieved_texts
    
    print(f"\n📚 Retrieved {len(retrieved_texts)} documents")
    for i, doc in enumerate(retrieved_texts, 1):
        print(f"   {i}. {doc[:80]}...")
    
    return state


def assess_document_relevance(state: CRAGState) -> CRAGState:
    """
    CRAG Step 1: Assess relevance of each retrieved document.
    Assigns confidence scores to determine if correction is needed.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    scores = []
    for doc in state["retrieved_docs"]:
        prompt = ChatPromptTemplate.from_template("""
        Question: {question}
        
        Document: {document}
        
        Rate the relevance of this document to answering the question.
        Respond with ONLY a number between 0.0 and 1.0:
        - 1.0 = Highly relevant, directly answers the question
        - 0.5 = Partially relevant, contains some useful information
        - 0.0 = Not relevant, doesn't help answer the question
        """)
        
        chain = prompt | llm | StrOutputParser()
        
        try:
            score = float(chain.invoke({
                "question": state["question"],
                "document": doc
            }).strip())
            scores.append(max(0.0, min(1.0, score)))
        except:
            scores.append(0.5)  # Default if parsing fails
    
    state["relevance_scores"] = scores
    state["overall_confidence"] = sum(scores) / len(scores) if scores else 0.0
    
    print(f"\n📊 Relevance Assessment:")
    for i, score in enumerate(scores, 1):
        print(f"   Doc {i}: {score:.2f}")
    print(f"   Overall Confidence: {state['overall_confidence']:.2f}")
    
    return state


def decide_correction_strategy(state: CRAGState) -> str:
    """
    CRAG Step 2: Decide if correction is needed.
    Routes to web search if confidence is low.
    """
    threshold = 0.6
    
    if state["overall_confidence"] < threshold:
        print(f"\n⚠️  Low confidence ({state['overall_confidence']:.2f} < {threshold})")
        print("   → Triggering web search for correction")
        return "web_search"
    else:
        print(f"\n✅ High confidence ({state['overall_confidence']:.2f} >= {threshold})")
        print("   → Proceeding with knowledge refinement")
        return "refine"


def web_search_correction(state: CRAGState) -> CRAGState:
    """
    CRAG Step 3a: Perform web search for additional/corrective information.
    """
    print(f"\n🌐 Performing web search...")
    
    try:
        search = DuckDuckGoSearchRun()
        results = search.run(state["question"])
        state["web_search_results"] = results
        state["correction_applied"] = True
        print(f"   Found: {len(results)} characters of web content")
    except Exception as e:
        print(f"   ⚠️  Web search failed: {e}")
        state["web_search_results"] = ""
        state["correction_applied"] = False
    
    return state


def refine_knowledge(state: CRAGState) -> CRAGState:
    """
    CRAG Step 3b: Refine knowledge by filtering irrelevant parts.
    Partitions documents into "knowledge strips" and filters.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    print(f"\n🔧 Refining knowledge...")
    
    # Filter documents based on relevance scores
    threshold = 0.4
    refined_docs = []
    
    for doc, score in zip(state["retrieved_docs"], state["relevance_scores"]):
        if score >= threshold:
            # Further refine by extracting only relevant sentences
            prompt = ChatPromptTemplate.from_template("""
            Question: {question}
            
            Document: {document}
            
            Extract ONLY the sentences that are directly relevant to answering the question.
            If nothing is relevant, respond with "NONE".
            """)
            
            chain = prompt | llm | StrOutputParser()
            refined = chain.invoke({
                "question": state["question"],
                "document": doc
            }).strip()
            
            if refined != "NONE":
                refined_docs.append(refined)
    
    state["refined_context"] = "\n\n".join(refined_docs)
    print(f"   Refined to {len(refined_docs)} relevant knowledge strips")
    
    return state


def generate_with_correction(state: CRAGState) -> CRAGState:
    """
    CRAG Step 4: Generate response using corrected/refined knowledge.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Combine refined context with web search results if available
    context_parts = []
    
    if state.get("refined_context"):
        context_parts.append(f"Retrieved Knowledge:\n{state['refined_context']}")
    
    if state.get("web_search_results"):
        context_parts.append(f"Web Search Results:\n{state['web_search_results'][:500]}")
    
    context = "\n\n---\n\n".join(context_parts)
    
    prompt = ChatPromptTemplate.from_template("""
    Answer the question using the provided context.
    
    Context:
    {context}
    
    Question: {question}
    
    Provide a clear, accurate answer. If the context is insufficient, say so.
    """)
    
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({
        "question": state["question"],
        "context": context
    })
    
    state["generation"] = response
    
    correction_note = " (with web search correction)" if state.get("correction_applied") else ""
    print(f"\n✍️  Generated response{correction_note}")
    
    return state


# ============================================================================
# 4. BUILD GRAPH
# ============================================================================

def build_crag_graph():
    """Build the Corrective RAG workflow graph."""
    workflow = StateGraph(CRAGState)
    
    # Add nodes
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("assess", assess_document_relevance)
    workflow.add_node("web_search", web_search_correction)
    workflow.add_node("refine", refine_knowledge)
    workflow.add_node("generate", generate_with_correction)
    
    # Add edges
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "assess")
    
    # Conditional routing based on confidence
    workflow.add_conditional_edges(
        "assess",
        decide_correction_strategy,
        {
            "web_search": "web_search",
            "refine": "refine"
        }
    )
    
    # Both paths lead to generation
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("refine", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()


# ============================================================================
# 5. MAIN EXECUTION
# ============================================================================

def main():
    """Run Corrective RAG examples."""
    print("\n🔧 Corrective RAG (CRAG) - November 2025")
    print("=" * 70)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not set")
        return
    
    # Build the graph
    app = build_crag_graph()
    
    # Example questions
    questions = [
        "When was LangGraph 1.0 released and what are its key features?",
        "What is the latest version of Python released in 2025?",  # Will trigger web search
        "Tell me about CrewAI v1.1.0 release"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'='*70}")
        print(f"EXAMPLE {i}: {question}")
        print('='*70)
        
        # Run the graph
        result = app.invoke({
            "question": question,
            "retrieved_docs": [],
            "relevance_scores": [],
            "overall_confidence": 0.0,
            "web_search_results": "",
            "refined_context": "",
            "generation": "",
            "correction_applied": False
        })
        
        # Display results
        print(f"\n📝 FINAL ANSWER:")
        print(f"{result['generation']}\n")
        print(f"📊 CRAG Metrics:")
        print(f"   Confidence: {result['overall_confidence']:.2f}")
        print(f"   Correction Applied: {result['correction_applied']}")
        print(f"   Knowledge Strips: {len(result['refined_context'].split('\\n\\n')) if result['refined_context'] else 0}")
    
    print("\n✅ All examples completed!")
    print("\n💡 CRAG Features Demonstrated:")
    print("   - Document relevance assessment")
    print("   - Confidence-based routing")
    print("   - Web search fallback")
    print("   - Knowledge refinement")
    print("   - Corrective generation")


if __name__ == "__main__":
    main()
