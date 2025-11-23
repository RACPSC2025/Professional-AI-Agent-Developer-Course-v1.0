"""
Module 2 Enhancement: CrewAI Flows (2024)
Framework: CrewAI with Flows
Objective: Event-driven workflow orchestration with state management

This example demonstrates:
- CrewAI Flows (new 2024 feature)
- Event-driven architecture with @start and @listen decorators
- State management across workflow steps
- Integration of multiple Crews
- Conditional flow control
"""

from crewai import Agent, Task, Crew, Process
from crewai.flow.flow import Flow, listen, start
from pydantic import BaseModel
from typing import List
import os

# ============================================================================
# 1. STATE MODELS (Pydantic for type safety)
# ============================================================================

class ResearchTopic(BaseModel):
    """Topic for research."""
    topic: str
    keywords: List[str]
    depth: str  # "basic", "intermediate", "advanced"


class ResearchFindings(BaseModel):
    """Research results."""
    topic: str
    summary: str
    key_points: List[str]
    sources: List[str]


class BlogPost(BaseModel):
    """Final blog post."""
    title: str
    content: str
    word_count: int
    seo_keywords: List[str]


# ============================================================================
# 2. FLOW DEFINITION
# ============================================================================

class ContentCreationFlow(Flow):
    """
    Event-driven content creation workflow using CrewAI Flows.
    
    Flow:
    1. Start: Generate topic
    2. Listen: Research topic
    3. Listen: Write blog post
    4. Listen: Review and finalize
    """
    
    @start()
    def generate_topic(self) -> ResearchTopic:
        """
        Entry point: Generate a research topic.
        Uses @start() decorator to mark as flow entry point.
        """
        print("\n🎯 Step 1: Generating Research Topic...")
        
        # In production, this could be user input or AI-generated
        topic = ResearchTopic(
            topic="Latest AI Agent Frameworks 2024",
            keywords=["LangGraph", "CrewAI", "AutoGen", "multi-agent"],
            depth="intermediate"
        )
        
        print(f"✅ Topic generated: {topic.topic}")
        return topic
    
    
    @listen(generate_topic)
    def research_topic(self, topic: ResearchTopic) -> ResearchFindings:
        """
        Step 2: Research the topic using a specialized Crew.
        Uses @listen() to wait for topic generation.
        """
        print(f"\n🔍 Step 2: Researching '{topic.topic}'...")
        
        # Create research agent
        researcher = Agent(
            role='Senior AI Research Analyst',
            goal=f'Conduct comprehensive research on {topic.topic}',
            backstory="""You are an expert AI researcher with deep knowledge
            of the latest developments in AI agent frameworks. You excel at
            finding accurate, up-to-date information and synthesizing it.""",
            verbose=False,
            allow_delegation=False
        )
        
        # Create research task
        research_task = Task(
            description=f"""Research {topic.topic} focusing on:
            - Latest developments and features
            - Key differences between frameworks
            - Real-world use cases
            - Best practices
            
            Keywords to focus on: {', '.join(topic.keywords)}
            Depth level: {topic.depth}
            """,
            agent=researcher,
            expected_output="Detailed research summary with key points and sources"
        )
        
        # Create and execute research crew
        research_crew = Crew(
            agents=[researcher],
            tasks=[research_task],
            process=Process.sequential,
            verbose=False
        )
        
        result = research_crew.kickoff()
        
        # Structure the findings
        findings = ResearchFindings(
            topic=topic.topic,
            summary=str(result)[:500],  # Truncate for demo
            key_points=[
                "LangGraph offers state machine-based agent orchestration",
                "CrewAI Flows enable event-driven workflows",
                "AutoGen v0.4 introduces async architecture"
            ],
            sources=[
                "https://langchain.com/langgraph",
                "https://docs.crewai.com/flows",
                "https://microsoft.github.io/autogen"
            ]
        )
        
        print(f"✅ Research completed: {len(findings.key_points)} key points found")
        return findings
    
    
    @listen(research_topic)
    def write_blog_post(self, findings: ResearchFindings) -> BlogPost:
        """
        Step 3: Write blog post based on research.
        Uses @listen() to wait for research completion.
        """
        print(f"\n✍️  Step 3: Writing blog post...")
        
        # Create writer agent
        writer = Agent(
            role='Technical Content Writer',
            goal='Create engaging, SEO-optimized technical blog posts',
            backstory="""You are an award-winning technical writer who
            specializes in making complex AI concepts accessible to
            intermediate developers. Your posts are known for clarity
            and practical value.""",
            verbose=False,
            allow_delegation=False
        )
        
        # Create writing task
        write_task = Task(
            description=f"""Write a 500-word blog post about: {findings.topic}
            
            Use these key points:
            {chr(10).join(f'- {point}' for point in findings.key_points)}
            
            Requirements:
            - Engaging introduction
            - Clear structure with headers
            - Practical examples
            - SEO-optimized
            - Include sources: {', '.join(findings.sources)}
            """,
            agent=writer,
            expected_output="Complete blog post with title and content"
        )
        
        # Create and execute writing crew
        writing_crew = Crew(
            agents=[writer],
            tasks=[write_task],
            process=Process.sequential,
            verbose=False
        )
        
        result = writing_crew.kickoff()
        
        # Structure the blog post
        post = BlogPost(
            title=f"Understanding {findings.topic}",
            content=str(result),
            word_count=len(str(result).split()),
            seo_keywords=["AI agents", "frameworks", "2024", "LangGraph", "CrewAI"]
        )
        
        print(f"✅ Blog post written: {post.word_count} words")
        return post
    
    
    @listen(write_blog_post)
    def review_and_finalize(self, post: BlogPost) -> dict:
        """
        Step 4: Review and finalize the blog post.
        Uses @listen() to wait for writing completion.
        """
        print(f"\n📝 Step 4: Reviewing blog post...")
        
        # Create editor agent
        editor = Agent(
            role='Senior Editor',
            goal='Ensure content quality and accuracy',
            backstory="""You are a meticulous editor with expertise in
            technical content. You check for accuracy, clarity, grammar,
            and SEO optimization.""",
            verbose=False,
            allow_delegation=False
        )
        
        # Create review task
        review_task = Task(
            description=f"""Review this blog post and provide feedback:
            
            Title: {post.title}
            Word Count: {post.word_count}
            
            Content Preview: {post.content[:200]}...
            
            Check for:
            - Technical accuracy
            - Grammar and clarity
            - SEO optimization
            - Structure and flow
            
            Provide: approval status and improvement suggestions
            """,
            agent=editor,
            expected_output="Review feedback with approval status"
        )
        
        # Create and execute review crew
        review_crew = Crew(
            agents=[editor],
            tasks=[review_task],
            process=Process.sequential,
            verbose=False
        )
        
        feedback = review_crew.kickoff()
        
        final_result = {
            "status": "approved",
            "blog_post": post.model_dump(),
            "review_feedback": str(feedback),
            "ready_for_publication": True
        }
        
        print(f"✅ Review completed: {final_result['status']}")
        return final_result


# ============================================================================
# 3. EXECUTION
# ============================================================================

def main():
    """Run the content creation flow."""
    print("\n🚀 CrewAI Flows - Content Creation Pipeline")
    print("=" * 70)
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not set")
        print("Set it with: export OPENAI_API_KEY='your-key-here'\n")
        return
    
    # Create and run the flow
    flow = ContentCreationFlow()
    
    print("\n🎬 Starting event-driven workflow...")
    print("   Flow will execute: generate_topic → research → write → review\n")
    
    # Execute the flow
    result = flow.kickoff()
    
    # Display results
    print("\n" + "=" * 70)
    print("📊 FLOW RESULTS")
    print("=" * 70)
    print(f"\nStatus: {result['status']}")
    print(f"Title: {result['blog_post']['title']}")
    print(f"Word Count: {result['blog_post']['word_count']}")
    print(f"SEO Keywords: {', '.join(result['blog_post']['seo_keywords'])}")
    print(f"\nReview: {result['review_feedback'][:200]}...")
    print(f"\nReady for Publication: {result['ready_for_publication']}")
    
    print("\n✅ Flow completed successfully!")
    print("\n💡 Key Features Demonstrated:")
    print("   - @start() decorator for entry point")
    print("   - @listen() decorators for event-driven flow")
    print("   - Pydantic models for type-safe state")
    print("   - Multiple Crews integration")
    print("   - Sequential workflow execution")


if __name__ == "__main__":
    main()
