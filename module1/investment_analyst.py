import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
import yfinance as yf
from langchain_community.tools import DuckDuckGoSearchRun

# Load environment variables
load_dotenv()

# 1. Define Tools
@tool
def get_stock_price(symbol: str):
    """Gets the current stock price for a given symbol (e.g., AAPL, MSFT)."""
    try:
        ticker = yf.Ticker(symbol)
        todays_data = ticker.history(period='1d')
        return f"Current price of {symbol}: ${todays_data['Close'].iloc[0]:.2f}"
    except Exception as e:
        return f"Error fetching stock price for {symbol}: {e}"

@tool
def get_company_news(query: str):
    """Searches for the latest news about a company or financial topic."""
    search = DuckDuckGoSearchRun()
    return search.run(query)

tools = [get_stock_price, get_company_news]

# 2. Initialize LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 3. Create Agent Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert AI Investment Analyst. 
    Your mission is to provide a comprehensive financial analysis for a given company.
    
    Follow this process:
    1. Get the current stock price.
    2. Search for the latest news and market sentiment.
    3. Combine the data to provide a recommendation (Buy/Sell/Hold) with a clear rationale.
    
    Be professional, data-driven, and concise."""),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# 4. Construct Agent
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 5. Run Agent (Example)
if __name__ == "__main__":
    print("--- AI Investment Analyst ---")
    company = input("Enter company symbol (e.g., AAPL, NVDA): ")
    if company:
        response = agent_executor.invoke({"input": f"Analyze {company} for me."})
        print("\n--- Analysis Report ---")
        print(response["output"])
