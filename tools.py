import os
import csv
import logging
from typing import Type
from langchain.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field
try:
    from duckduckgo_search import DDGS
except ImportError:
    logging.warning("duckduckgo-search not installed. Web search functionality will be limited.")
    DDGS = None

try:
    import wikipedia
except ImportError:
    logging.warning("wikipedia not installed. Wikipedia search functionality will be unavailable.")
    wikipedia = None


class SearchInput(BaseModel):
    """Input schema for search tool"""
    query: str = Field(description="Search query to find information about")


class WikipediaInput(BaseModel):
    """Input schema for Wikipedia tool"""
    query: str = Field(description="Wikipedia search query")


class SaveInput(BaseModel):
    """Input schema for save tool"""
    content: str = Field(description="Content to save to file")


class DuckDuckGoSearchTool(BaseTool):
    """Tool for searching the web using DuckDuckGo"""
    name: str = "search"
    description: str = "Search the web for current information using DuckDuckGo. Use this for recent news, current events, and general web information."
    args_schema: Type[BaseModel] = SearchInput

    def _run(
        self,
        query: str,
        run_manager: CallbackManagerForToolRun = None,
    ) -> str:
        """Execute the search"""
        if not DDGS:
            return "Error: DuckDuckGo search is not available. Please install duckduckgo-search package."
        
        try:
            logging.debug(f"Searching DuckDuckGo for: {query}")
            
            # Create DDGS instance and perform search
            ddgs = DDGS()
            results = list(ddgs.text(query, max_results=5))
            
            if not results:
                return f"No search results found for query: {query}"
            
            # Format results
            formatted_results = []
            for i, result in enumerate(results, 1):
                title = result.get('title', 'No title')
                body = result.get('body', 'No description')
                href = result.get('href', 'No URL')
                
                formatted_results.append(
                    f"{i}. **{title}**\n"
                    f"   {body}\n"
                    f"   Source: {href}\n"
                )
            
            search_summary = f"Search results for '{query}':\n\n" + "\n".join(formatted_results)
            logging.debug(f"DuckDuckGo search completed. Found {len(results)} results.")
            
            return search_summary
            
        except Exception as e:
            error_msg = f"Error searching DuckDuckGo: {str(e)}"
            logging.error(error_msg)
            return error_msg


class WikipediaTool(BaseTool):
    """Tool for searching Wikipedia"""
    name: str = "wikipedia"
    description: str = "Search Wikipedia for encyclopedic information. Best for factual, historical, and scientific information."
    args_schema: Type[BaseModel] = WikipediaInput

    def _run(
        self,
        query: str,
        run_manager: CallbackManagerForToolRun = None,
    ) -> str:
        """Execute Wikipedia search"""
        if not wikipedia:
            return "Error: Wikipedia search is not available. Please install wikipedia package."
        
        try:
            logging.debug(f"Searching Wikipedia for: {query}")
            
            # Set language to English
            wikipedia.set_lang("en")
            
            # Search for pages
            search_results = wikipedia.search(query, results=3)
            
            if not search_results:
                return f"No Wikipedia articles found for query: {query}"
            
            # Get summary of the first result
            try:
                page_title = search_results[0]
                summary = wikipedia.summary(page_title, sentences=3)
                page_url = wikipedia.page(page_title).url
                
                result = (
                    f"Wikipedia search results for '{query}':\n\n"
                    f"**{page_title}**\n"
                    f"{summary}\n"
                    f"Source: {page_url}\n\n"
                )
                
                # Add additional search results if available
                if len(search_results) > 1:
                    result += "Additional related articles:\n"
                    for title in search_results[1:]:
                        try:
                            page_url = wikipedia.page(title).url
                            result += f"- {title}: {page_url}\n"
                        except:
                            result += f"- {title}\n"
                
                logging.debug(f"Wikipedia search completed for: {page_title}")
                return result
                
            except wikipedia.exceptions.DisambiguationError as e:
                # Handle disambiguation
                page_title = e.options[0]
                summary = wikipedia.summary(page_title, sentences=3)
                page_url = wikipedia.page(page_title).url
                
                result = (
                    f"Wikipedia search results for '{query}' (disambiguation resolved):\n\n"
                    f"**{page_title}**\n"
                    f"{summary}\n"
                    f"Source: {page_url}\n"
                )
                return result
                
            except wikipedia.exceptions.PageError:
                return f"Wikipedia page not found for: {search_results[0]}"
                
        except Exception as e:
            error_msg = f"Error searching Wikipedia: {str(e)}"
            logging.error(error_msg)
            return error_msg


class SaveTextTool(BaseTool):
    """Tool for saving research content to CSV file"""
    name: str = "save_text_to_file"
    description: str = "Save research content and findings to the data.csv file for persistence."
    args_schema: Type[BaseModel] = SaveInput

    def _run(
        self,
        content: str,
        run_manager: CallbackManagerForToolRun = None,
    ) -> str:
        """Save content to file"""
        try:
            logging.debug(f"Saving content to data.csv: {content[:100]}...")
            
            # For this tool, we'll just acknowledge the save request
            # The actual saving will be handled by the main application
            # after the agent completes its research
            
            return f"Content marked for saving to data.csv: {content[:100]}..."
            
        except Exception as e:
            error_msg = f"Error saving content: {str(e)}"
            logging.error(error_msg)
            return error_msg


# Create tool instances
search_tool = DuckDuckGoSearchTool()
wiki_tool = WikipediaTool()
save_tool = SaveTextTool()

# Export tools for use in the main application
__all__ = ['search_tool', 'wiki_tool', 'save_tool']
