#did not find this yf_news_tool.py used anywhere in the application. leaving this in the original_files folder

from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsInput
from crewai_tools import tool

@tool
def yahoo_news_tool(stock_symbol):
    """
    Perform a comprehensive technical analysis on the given stock symbol.
    
    Args:
        stock_symbol (str): The stock symbol to analyze.
        period (str): The time period for analysis. Default is "1y" (1 year).
    
    Returns:
        dict: A dictionary with the detailed technical analysis results.
    """
    return YahooFinanceNewsInput(query=stock_symbol)