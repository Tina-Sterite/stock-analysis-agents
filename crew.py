import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, ScrapeWebsiteTool, WebsiteSearchTool
from tools.sentiment_analysis_tool import reddit_sentiment_analysis
from tools.yf_tech_analysis_tool import yf_tech_analysis
from tools.yf_fundamental_analysis_tool import yf_fundamental_analysis
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import sys
from contextlib import redirect_stdout
from datetime import datetime
import markdown2
from weasyprint import HTML
from weasyprint.text.fonts import FontConfiguration
import agentops

# Environment Variables
load_dotenv()

# Set USER_AGENT environment variable if not already set
USER_AGENT = os.getenv("USER_AGENT")

os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")
os.environ["REDDIT_CLIENT_ID"] = os.getenv("REDDIT_CLIENT_ID")
os.environ["REDDIT_CLIENT_SECRET"] = os.getenv("REDDIT_CLIENT_SECRET")
os.environ["REDDIT_USER_AGENT"] = os.getenv("REDDIT_USER_AGENT")
os.environ["AGENTOPS_API_KEY"] = os.getenv("AGENTOPS_API_KEY")
agentops_api_key=os.getenv("AGENTOPS_API_KEY")

# Model Selection
def initialize_llm(model_option, openai_api_key, groq_api_key):
    if model_option == 'OpenAI GPT-4o Mini':
        return ChatOpenAI(openai_api_key=openai_api_key, model='gpt-4o-mini', temperature=0.1)
    elif model_option == 'Llama 3 8B':
        return ChatGroq(groq_api_key=groq_api_key, model='llama3-8b-8192', temperature=0.1)
    elif model_option == 'Llama 3.1 70B':
        return ChatGroq(groq_api_key=groq_api_key, model='llama-3.1-70b-versatile', temperature=0.1)
    elif model_option == 'Llama 3.1 8B':
        return ChatGroq(groq_api_key=groq_api_key, model='llama-3.1-8b-instant', temperature=0.1)
    else:
        raise ValueError("Invalid model option selected")

def create_crew(stock_symbol, model_option, openai_api_key, groq_api_key, console_output_file_path):
    agentops.init(agentops_api_key)
    llm = initialize_llm(model_option, openai_api_key, groq_api_key)
    # Tools Initialization
    reddit_tool = reddit_sentiment_analysis
    serper_tool = SerperDevTool()
    yf_tech_tool = yf_tech_analysis
    yf_fundamental_tool = yf_fundamental_analysis
    
    # Agents Definitions
    researcher = Agent(
        role='Senior Stock Market Researcher',
        goal='Gather and analyze comprehensive data about {stock_symbol}',
        verbose=True,
        memory=True,
        backstory="With a Ph.D. in Financial Economics and 15 years of experience in equity research, you're known for your meticulous data collection and insightful analysis.",
        tools=[reddit_tool, serper_tool, YahooFinanceNewsTool()],
        llm=llm
    )

    technical_analyst = Agent(
        role='Expert Technical Analyst',
        goal='Perform an in-depth technical analysis on {stock_symbol}',
        verbose=True,
        memory=True,
        backstory="As a Chartered Market Technician (CMT) with 15 years of experience, you have a keen eye for chart patterns and market trends.",
        tools=[yf_tech_tool],
        llm=llm
    )

    fundamental_analyst = Agent(
        role='Senior Fundamental Analyst',
        goal='Conduct a comprehensive fundamental analysis of {stock_symbol}',
        verbose=True,
        memory=True,
        backstory="With a CFA charter and 15 years of experience in value investing, you dissect financial statements and identify key value drivers.",
        tools=[yf_fundamental_tool],
        llm=llm
    )

    reporter = Agent(
        role='Chief Investment Strategist',
        goal='Synthesize all analyses to create a definitive investment report on {stock_symbol}',
        verbose=True,
        memory=True,
        backstory="As a seasoned investment strategist with 20 years of experience, you weave complex financial data into compelling investment narratives.",
        tools=[reddit_tool, serper_tool, yf_fundamental_tool, yf_tech_tool, YahooFinanceNewsTool()],
        llm=llm
    )

    # Task Definitions
    research_task = Task(
        description=(
            "Conduct research on {stock_symbol}. Your analysis should include:\n"
            "1. Current stock price and historical performance (5 years).\n"
            "2. Key financial metrics (P/E, EPS growth, revenue growth, margins).\n"
            "3. Recent news and press releases (1 month).\n"
            "4. Analyst ratings and price targets (min 3 analysts).\n"
            "5. Reddit sentiment analysis (100 posts).\n"
            "6. Major institutional holders and recent changes.\n"
            "7. Competitive landscape and {stock_symbol}'s market share.\n"
            "Use reputable financial websites for data."
        ),
        expected_output='A detailed 150-word research report with data sources and brief analysis.',
        agent=researcher
    )

    technical_analysis_task = Task(
        description=(
            "Perform technical analysis on {stock_symbol}. Include:\n"
            "1. 50-day and 200-day moving averages (1 year).\n"
            "2. Key support and resistance levels (3 each).\n"
            "3. RSI and MACD indicators.\n"
            "4. Volume analysis (3 months).\n"
            "5. Significant chart patterns (6 months).\n"
            "6. Fibonacci retracement levels.\n"
            "7. Comparison with sector's average.\n"
            "Use the yf_tech_analysis tool for data."
        ),
        expected_output='A 100-word technical analysis report with buy/sell/hold signals and annotated charts.',
        agent=technical_analyst
    )

    fundamental_analysis_task = Task(
        description=(
            "Conduct fundamental analysis of {stock_symbol}. Include:\n"
            "1. Review last 3 years of financial statements.\n"
            "2. Key ratios (P/E, P/B, P/S, PEG, Debt-to-Equity, etc.).\n"
            "3. Comparison with main competitors and industry averages.\n"
            "4. Revenue and earnings growth trends.\n"
            "5. Management effectiveness (ROE, capital allocation).\n"
            "6. Competitive advantages and market position.\n"
            "7. Growth catalysts and risks (2-3 years).\n"
            "8. DCF valuation model with assumptions.\n"
            "Use yf_fundamental_analysis tool for data."
        ),
        expected_output='A 100-word fundamental analysis report with buy/hold/sell recommendation and key metrics summary.',
        agent=fundamental_analyst
    )

    report_task = Task(
        description=(
            "Create an investment report on {stock_symbol}. Include:\n"
            "1. Executive Summary: Investment recommendation.\n"
            "2. Company Snapshot: Key facts.\n"
            "3. Financial Highlights: Top metrics and peer comparison.\n"
            "4. Technical Analysis: Key findings.\n"
            "5. Fundamental Analysis: Top strengths and concerns.\n"
            "6. Risk and Opportunity: Major risk and growth catalyst.\n"
            "7. Reddit Sentiment: Key takeaway from sentiment analysis, including the number of positive, negative and neutral comments and total comments.\n"
            "8. Investment Thesis: Bull and bear cases.\n"
            "9. Price Target: 12-month forecast.\n"
        ),
        expected_output='A 600-word investment report with clear sections, key insights.',
        agent=reporter
    )

    # Create the crew and kickoff
    crew = Crew(
        agents=[researcher, technical_analyst, fundamental_analyst, reporter],
        tasks=[research_task, technical_analysis_task, fundamental_analysis_task, report_task],
        process=Process.sequential,
        cache=True
    )

    # Use contextlib.redirect_stdout to capture both the result and the console output
    with open(console_output_file_path, 'w', encoding='utf-8') as f:
        with redirect_stdout(f):
            result = crew.kickoff(inputs={'stock_symbol': stock_symbol})
            agentops.end_session('Success')
                
    # Append usage details to the log file
    with open(console_output_file_path, 'a', encoding='utf-8') as f:
        f.write('\nUsage Details:\n')
        f.write(f'Prompt Tokens: {result.token_usage["prompt_tokens"]}\n')
        f.write(f'Completion Tokens: {result.token_usage["completion_tokens"]}\n')
        f.write(f'Total Tokens: {result.token_usage["total_tokens"]}\n')
    
    #  response_content = result.choices[0].message.content
    
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    # Save the result to a separate file
    crew_result_file_path = f"./crew_results/{stock_symbol}_Stock_Report_{current_date}.md"
    with open(crew_result_file_path, 'w', encoding='utf-8') as file:
        file.write(str(result))
        
    
    # Convert the Markdown content to HTML
    html_content = markdown2.markdown_path(crew_result_file_path)

    # Save the HTML content to a temporary HTML file
    crew_result_file_path_html = f"./crew_results/{stock_symbol}_Stock_Report_{current_date}.html"
    with open(crew_result_file_path_html, 'w', encoding='utf-8') as file:
        file.write(html_content)

    # Create a FontConfiguration instance
    font_config = FontConfiguration()

    # Convert the HTML file to PDF
    crew_result_file_path_pdf = f"./crew_results/{stock_symbol}_Stock_Report_{current_date}.pdf"
    HTML(crew_result_file_path_html).write_pdf(crew_result_file_path_pdf, font_config=font_config)
        
    return crew_result_file_path, crew_result_file_path_pdf