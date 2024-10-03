import json
import os

import requests
from langchain.tools import tool
import logging

# Import the logger
from logger import logger

class SearchTools():
  @tool("Search the internet")
  def search_internet(query):
    """Useful to search the internet 
    about a a given topic and return relevant results"""
    try:
      top_result_to_return = 4
      url = "https://google.serper.dev/search"
      payload = json.dumps({"q": query})
      headers = {
          'X-API-KEY': os.environ['SERPER_API_KEY'],
          'content-type': 'application/json'
      }
      #make the API call
      response = requests.request("POST", url, headers=headers, data=payload)
      
      # Log token usage
      logger.info(f"Serper API Token Used: {os.environ['SERPER_API_KEY'][:5]}...")
      
      results = response.json()['organic']
      string = []
      for result in results[:top_result_to_return]:
        try:
          string.append('\n'.join([
              f"Title: {result['title']}", f"Link: {result['link']}",
              f"Snippet: {result['snippet']}", "\n-----------------"
         ]))
        except KeyError:
         next

      return '\n'.join(string)
    except Exception as e:
      logger.error(f"Error in search_internet: {str(e)}")
      return f"An error occurred during the search: {str(e)}"

  @tool("Search news on the internet")
  def search_news(query):
    """Useful to search news about a company, stock or any other
    topic and return relevant results"""""
    try:
      top_result_to_return = 4
      url = "https://google.serper.dev/news"
      payload = json.dumps({"q": query})
      headers = {
          'X-API-KEY': os.environ['SERPER_API_KEY'],
         'content-type': 'application/json'
     }
      #make the API call
      response = requests.request("POST", url, headers=headers, data=payload)
    
      # Log token usage
      logger.info(f"Serper API Token Used: {os.environ['SERPER_API_KEY'][:5]}...")
      
      results = response.json()['news']
      string = []
      for result in results[:top_result_to_return]:
        try:
          string.append('\n'.join([
              f"Title: {result['title']}", f"Link: {result['link']}",
              f"Snippet: {result['snippet']}", "\n-----------------"
          ]))
        except KeyError:
          next

      return '\n'.join(string)
    except Exception as e:
      logger.error(f"Error in search_news: {str(e)}")
      return f"An error occurred during the search: {str(e)}"