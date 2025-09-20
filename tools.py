import os
import csv
import logging
import requests
from typing import Type, Optional, Any
from langchain.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain.tools import Tool
from pydantic import BaseModel, Field
from bs4 import BeautifulSoup
import requests
from bs4 import BeautifulSoup
try:
    from duckduckgo_search import DDGS
except ImportError:
    logging.warning(
        "duckduckgo-search not installed. Web search functionality will be limited."
    )
    DDGS = None

try:
    import wikipedia
except ImportError:
    logging.warning(
        "wikipedia not installed. Wikipedia search functionality will be unavailable."
    )
    wikipedia = None


class BookSearchInput(BaseModel):
    """Input schema for book search tool"""
    query: str = Field(description="Book title, author, or description to search for")


class BookSearchTool(BaseTool):
    """Tool for searching books and extracting title, author, and first publication year"""
    name: str = "book_search"
    description: str = "Search for books and return title, author, and first year of publishing. Use this to find specific book information."
    args_schema: Type[BaseModel] = BookSearchInput

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute the book search"""
        try:
            logging.debug(f"Searching for book: {query}")
            
            # First try Wikipedia for reliable book information
            book_info = self._search_wikipedia_books(query)
            if book_info:
                return book_info
                
            # If Wikipedia doesn't have good results, try web search
            book_info = self._search_web_books(query)
            if book_info:
                return book_info
                
            return f"No reliable book information found for query: {query}"

        except Exception as e:
            error_msg = f"Error searching for book: {str(e)}"
            logging.error(error_msg)
            return error_msg

    def _search_wikipedia_books(self, query: str) -> Optional[str]:
        """Search Wikipedia for book information"""
        if not wikipedia:
            return None
            
        try:
            wikipedia.set_lang("en")
            
            # Add "(book)" or "(novel)" to improve book search results
            search_queries = [
                f"{query} book",
                f"{query} novel", 
                query
            ]
            
            for search_query in search_queries:
                search_results = wikipedia.search(search_query, results=5)
                
                for result_title in search_results:
                    try:
                        page = wikipedia.page(result_title)
                        content = page.content.lower()
                        
                        # Check if this page is likely about a book
                        book_indicators = [
                            "is a novel", "is a book", "published", "author", 
                            "first published", "publication", "publisher", 
                            "isbn", "fiction", "non-fiction"
                        ]
                        
                        if any(indicator in content for indicator in book_indicators):
                            return self._extract_book_info_from_wikipedia(page, query)
                            
                    except (wikipedia.exceptions.DisambiguationError, 
                            wikipedia.exceptions.PageError):
                        continue
                        
        except Exception as e:
            logging.error(f"Wikipedia search error: {e}")
            
        return None

    def _extract_book_info_from_wikipedia(self, page, original_query: str) -> str:
        """Extract book information from Wikipedia page"""
        try:
            title = page.title
            content = page.content
            
            # Extract author - look for common patterns
            author = "Unknown Author"
            author_patterns = [
                r"written by ([^.\n]+)",
                r"authored by ([^.\n]+)", 
                r"by ([A-Z][a-z]+ [A-Z][a-z]+)",
                r"([A-Z][a-z]+ [A-Z][a-z]+) is the author",
                r"([A-Z][a-z]+ [A-Z][a-z]+) wrote"
            ]
            
            import re
            for pattern in author_patterns:
                match = re.search(pattern, content)
                if match:
                    author = match.group(1).strip()
                    # Clean up common suffixes
                    author = re.sub(r'\s+(is a|was a|wrote|authored).*', '', author)
                    break
            
            # Extract publication year
            year = "Unknown Year"
            year_patterns = [
                r"first published in (\d{4})",
                r"published in (\d{4})",
                r"(\d{4}) publication",
                r"released in (\d{4})",
                r"\b(\d{4})\b"
            ]
            
            for pattern in year_patterns:
                matches = re.findall(pattern, content)
                if matches:
                    # Get the earliest year that makes sense for a book (after 1000 AD)
                    years = [int(year) for year in matches if 1000 <= int(year) <= 2024]
                    if years:
                        year = str(min(years))
                        break
            
            result = f"""Book found: "{title}"
Author: {author}
First Year Published: {year}
Source: Wikipedia - {page.url}

This information was extracted from Wikipedia's article about the book."""
            
            logging.debug(f"Extracted book info: {title} by {author} ({year})")
            return result
            
        except Exception as e:
            logging.error(f"Error extracting book info: {e}")
            return None

    def _search_web_books(self, query: str) -> Optional[str]:
        """Search the web for book information using DuckDuckGo"""
        if not DDGS:
            return None
            
        try:
            ddgs = DDGS()
            book_query = f"{query} book author publication year"
            results = list(ddgs.text(book_query, max_results=5))
            
            for result in results:
                title = result.get('title', '')
                body = result.get('body', '')
                href = result.get('href', '')
                
                # Check if result is book-related
                book_keywords = ['book', 'novel', 'author', 'published', 'publication']
                if any(keyword in title.lower() or keyword in body.lower() for keyword in book_keywords):
                    
                    # Try to extract book information from the result
                    book_info = self._extract_book_info_from_text(title + " " + body, query, href)
                    if book_info:
                        return book_info
                        
        except Exception as e:
            logging.error(f"Web search error: {e}")
            
        return None

    def _extract_book_info_from_text(self, text: str, original_query: str, source_url: str) -> Optional[str]:
        """Extract book information from web search text"""
        import re
        
        # Extract title - often in quotes or after "book" 
        title = original_query  # Default to the search query
        title_patterns = [
            r'"([^"]+)"',
            r"'([^']+)'",
            r"book (?:titled |called )?([A-Za-z ]+)",
            r"novel (?:titled |called )?([A-Za-z ]+)"
        ]
        
        for pattern in title_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                potential_title = match.group(1).strip()
                if len(potential_title) > 3:  # Reasonable title length
                    title = potential_title
                    break
        
        # Extract author
        author = "Unknown Author"
        author_patterns = [
            r"by ([A-Z][a-z]+ [A-Z][a-z]+)",
            r"author ([A-Z][a-z]+ [A-Z][a-z]+)",
            r"written by ([A-Z][a-z]+ [A-Z][a-z]+)"
        ]
        
        for pattern in author_patterns:
            match = re.search(pattern, text)
            if match:
                author = match.group(1).strip()
                break
        
        # Extract year
        year = "Unknown Year"
        year_matches = re.findall(r'\b(\d{4})\b', text)
        if year_matches:
            # Get years that could be publication dates
            valid_years = [int(y) for y in year_matches if 1000 <= int(y) <= 2024]
            if valid_years:
                year = str(min(valid_years))
        
        result = f"""Book found: "{title}"
Author: {author}
First Year Published: {year}
Source: Web search - {source_url}

This information was found through web search and may need verification."""

        return result


# Create tool instance
book_search_tool = BookSearchTool()

# Export tools for use in the main application
__all__ = ['book_search_tool']