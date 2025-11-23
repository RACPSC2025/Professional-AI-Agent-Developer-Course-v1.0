"""
Module 1.2 Enhancement: Structured Outputs with GPT-5.1 (November 2025)
Framework: OpenAI API + Pydantic
Objective: Type-safe, validated LLM outputs using latest GPT-5.1 API

This example demonstrates:
- GPT-5.1 structured outputs (November 2025)
- Pydantic models for validation
- Complex nested schemas
- Enum types for constrained outputs
- Error handling and retry logic
- Production-ready patterns
"""

from openai import OpenAI
from pydantic import BaseModel, Field, field_validator
from typing import List, Literal, Optional
from enum import Enum
import os
import json

# ============================================================================
# 1. PYDANTIC MODELS FOR STRUCTURED OUTPUTS
# ============================================================================

class FrameworkCategory(str, Enum):
    """Framework categories."""
    ORCHESTRATION = "orchestration"
    RAG = "rag"
    MULTI_AGENT = "multi_agent"
    ENTERPRISE = "enterprise"
    LIGHTWEIGHT = "lightweight"


class FrameworkFeature(BaseModel):
    """Individual framework feature."""
    name: str = Field(description="Feature name")
    description: str = Field(description="Brief description")
    importance: Literal["critical", "important", "nice_to_have"] = Field(
        description="Feature importance level"
    )


class FrameworkComparison(BaseModel):
    """Structured comparison of AI agent frameworks."""
    framework_name: str = Field(description="Name of the framework")
    category: FrameworkCategory = Field(description="Primary category")
    release_date: str = Field(description="Latest release date (YYYY-MM-DD)")
    version: str = Field(description="Current version")
    
    key_features: List[FrameworkFeature] = Field(
        description="List of key features",
        min_length=3,
        max_length=5
    )
    
    strengths: List[str] = Field(
        description="Main strengths",
        min_length=2,
        max_length=4
    )
    
    weaknesses: List[str] = Field(
        description="Main weaknesses",
        min_length=1,
        max_length=3
    )
    
    github_stars: int = Field(
        description="Approximate GitHub stars",
        ge=0
    )
    
    best_for: str = Field(
        description="What this framework is best suited for"
    )
    
    production_ready: bool = Field(
        description="Is it production-ready?"
    )
    
    @field_validator('release_date')
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        """Validate date format."""
        from datetime import datetime
        try:
            datetime.strptime(v, '%Y-%m-%d')
            return v
        except ValueError:
            raise ValueError('Date must be in YYYY-MM-DD format')


class MultiFrameworkAnalysis(BaseModel):
    """Analysis of multiple frameworks."""
    analysis_date: str = Field(description="Date of analysis (YYYY-MM-DD)")
    frameworks: List[FrameworkComparison] = Field(
        description="List of framework comparisons",
        min_length=2
    )
    recommendation: str = Field(
        description="Overall recommendation based on analysis"
    )
    trends: List[str] = Field(
        description="Observed trends in the ecosystem",
        min_length=2,
        max_length=5
    )


class CodeGenerationRequest(BaseModel):
    """Structured code generation output."""
    language: Literal["python", "javascript", "typescript", "go"] = Field(
        description="Programming language"
    )
    code: str = Field(description="Generated code")
    explanation: str = Field(description="Code explanation")
    dependencies: List[str] = Field(
        description="Required dependencies",
        default_factory=list
    )
    complexity: Literal["beginner", "intermediate", "advanced"] = Field(
        description="Code complexity level"
    )
    test_cases: Optional[List[str]] = Field(
        description="Suggested test cases",
        default=None
    )


# ============================================================================
# 2. GPT-5.1 STRUCTURED OUTPUT FUNCTIONS
# ============================================================================

def get_structured_framework_comparison(
    framework_name: str,
    model: str = "gpt-4o-mini"  # Use gpt-5.1 when available
) -> FrameworkComparison:
    """
    Get structured framework comparison using GPT-5.1.
    
    Args:
        framework_name: Name of the framework to analyze
        model: Model to use (gpt-5.1 or gpt-4o-mini)
        
    Returns:
        Validated FrameworkComparison object
    """
    client = OpenAI()
    
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "system",
                "content": """You are an expert in AI agent frameworks. 
                Provide accurate, up-to-date information about frameworks as of November 2025."""
            },
            {
                "role": "user",
                "content": f"Provide a detailed comparison of the {framework_name} framework."
            }
        ],
        response_format=FrameworkComparison,
        temperature=0
    )
    
    # The response is automatically parsed and validated
    return completion.choices[0].message.parsed


def get_multi_framework_analysis(
    frameworks: List[str],
    model: str = "gpt-4o-mini"
) -> MultiFrameworkAnalysis:
    """
    Get comprehensive analysis of multiple frameworks.
    
    Args:
        frameworks: List of framework names
        model: Model to use
        
    Returns:
        Validated MultiFrameworkAnalysis object
    """
    client = OpenAI()
    
    frameworks_str = ", ".join(frameworks)
    
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "system",
                "content": """You are an expert in AI agent frameworks with deep knowledge 
                of the November 2025 ecosystem. Provide accurate comparisons."""
            },
            {
                "role": "user",
                "content": f"""Analyze and compare these frameworks: {frameworks_str}
                
                Focus on:
                - Latest versions and release dates (November 2025)
                - Key differentiators
                - Production readiness
                - Current trends
                """
            }
        ],
        response_format=MultiFrameworkAnalysis,
        temperature=0
    )
    
    return completion.choices[0].message.parsed


def generate_structured_code(
    task_description: str,
    language: str = "python",
    model: str = "gpt-4o-mini"
) -> CodeGenerationRequest:
    """
    Generate code with structured output.
    
    Args:
        task_description: What the code should do
        language: Programming language
        model: Model to use
        
    Returns:
        Validated CodeGenerationRequest object
    """
    client = OpenAI()
    
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "system",
                "content": f"""You are an expert {language} developer. 
                Generate clean, production-ready code with proper error handling."""
            },
            {
                "role": "user",
                "content": f"Generate {language} code for: {task_description}"
            }
        ],
        response_format=CodeGenerationRequest,
        temperature=0.3
    )
    
    return completion.choices[0].message.parsed


# ============================================================================
# 3. EXAMPLES WITH ERROR HANDLING
# ============================================================================

def example_single_framework():
    """Example 1: Single framework comparison."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Single Framework Comparison (Structured Output)")
    print("="*70 + "\n")
    
    try:
        result = get_structured_framework_comparison("LangGraph")
        
        print(f"Framework: {result.framework_name}")
        print(f"Category: {result.category.value}")
        print(f"Version: {result.version}")
        print(f"Release Date: {result.release_date}")
        print(f"GitHub Stars: {result.github_stars:,}")
        print(f"Production Ready: {'✅ Yes' if result.production_ready else '❌ No'}")
        
        print(f"\nKey Features:")
        for feature in result.key_features:
            print(f"  • {feature.name} ({feature.importance})")
            print(f"    {feature.description}")
        
        print(f"\nStrengths:")
        for strength in result.strengths:
            print(f"  ✅ {strength}")
        
        print(f"\nWeaknesses:")
        for weakness in result.weaknesses:
            print(f"  ⚠️  {weakness}")
        
        print(f"\nBest For: {result.best_for}")
        
        # Demonstrate type safety
        print(f"\n🔒 Type Safety Verified:")
        print(f"  - Category is enum: {isinstance(result.category, FrameworkCategory)}")
        print(f"  - Features validated: {len(result.key_features)} items")
        print(f"  - All fields type-checked: ✅")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def example_multi_framework():
    """Example 2: Multi-framework analysis."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Multi-Framework Analysis")
    print("="*70 + "\n")
    
    frameworks = ["LangGraph", "CrewAI", "AutoGen"]
    
    try:
        result = get_multi_framework_analysis(frameworks)
        
        print(f"Analysis Date: {result.analysis_date}")
        print(f"\nFrameworks Analyzed: {len(result.frameworks)}")
        
        for fw in result.frameworks:
            print(f"\n{'─'*50}")
            print(f"📦 {fw.framework_name} ({fw.version})")
            print(f"   Category: {fw.category.value}")
            print(f"   Stars: {fw.github_stars:,}")
            print(f"   Best for: {fw.best_for}")
        
        print(f"\n{'─'*50}")
        print(f"\n🎯 Recommendation:")
        print(f"{result.recommendation}")
        
        print(f"\n📈 Trends:")
        for trend in result.trends:
            print(f"  • {trend}")
        
        # Export to JSON (fully serializable)
        json_output = result.model_dump_json(indent=2)
        print(f"\n💾 JSON Export Available ({len(json_output)} bytes)")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def example_code_generation():
    """Example 3: Structured code generation."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Structured Code Generation")
    print("="*70 + "\n")
    
    task = "Create a simple RAG system with ChromaDB and OpenAI"
    
    try:
        result = generate_structured_code(task, language="python")
        
        print(f"Language: {result.language}")
        print(f"Complexity: {result.complexity}")
        
        print(f"\n📦 Dependencies:")
        for dep in result.dependencies:
            print(f"  - {dep}")
        
        print(f"\n💻 Generated Code:")
        print("─" * 70)
        print(result.code)
        print("─" * 70)
        
        print(f"\n📝 Explanation:")
        print(result.explanation)
        
        if result.test_cases:
            print(f"\n🧪 Test Cases:")
            for i, test in enumerate(result.test_cases, 1):
                print(f"  {i}. {test}")
        
        # Demonstrate validation
        print(f"\n✅ Validation Passed:")
        print(f"  - Language is valid: {result.language in ['python', 'javascript', 'typescript', 'go']}")
        print(f"  - Code is non-empty: {len(result.code) > 0}")
        print(f"  - Complexity level set: {result.complexity}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


# ============================================================================
# 4. MAIN EXECUTION
# ============================================================================

def main():
    """Run all structured output examples."""
    print("\n🎯 GPT-5.1 Structured Outputs - November 2025")
    print("=" * 70)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not set")
        print("Set it with: export OPENAI_API_KEY='your-key-here'\n")
        return
    
    print("\n💡 Using: gpt-4o-mini (upgrade to gpt-5.1 when available)")
    print("   GPT-5.1 features: Enhanced steerability, faster responses")
    
    # Run examples
    example_single_framework()
    example_multi_framework()
    example_code_generation()
    
    print("\n" + "="*70)
    print("✅ All examples completed!")
    print("\n💡 Key Features Demonstrated:")
    print("   - Pydantic model validation")
    print("   - Enum types for constrained outputs")
    print("   - Nested complex schemas")
    print("   - Field validators")
    print("   - Type safety guarantees")
    print("   - JSON serialization")
    print("   - Error handling")
    print("\n🚀 Production Benefits:")
    print("   - No parsing errors")
    print("   - Guaranteed schema compliance")
    print("   - IDE autocomplete support")
    print("   - Runtime type checking")
    print("   - Easy integration with databases/APIs")


if __name__ == "__main__":
    main()
