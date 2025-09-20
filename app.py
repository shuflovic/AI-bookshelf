import os
import csv
import logging
from flask import Flask, send_file, jsonify, request
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import book_search_tool

# Configure logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-for-research-assistant")

# Load environment variables
load_dotenv()

class BookResponse(BaseModel):
    """Structured response model for book search output"""
    title: str
    author: str
    first_year_published: str
    search_query: str

def initialize_llm():
    """Initialize the AI model with support for multiple providers"""
    google_key = os.getenv('GEMINI_API_KEY')
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')

    if google_key:
        try:
            logging.info("Initializing Google Gemini")
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=google_key,
                temperature=0.7
            )
            return llm
        except Exception as e:
            logging.error(f"Error initializing Google Gemini: {e}")
            if anthropic_key:
                logging.info("Trying Anthropic Claude")

    if anthropic_key:
        try:
            logging.info("Initializing Anthropic Claude")
            llm = ChatAnthropic(
                model="claude-3-5-sonnet-20241022",
                anthropic_api_key=anthropic_key
            )
            return llm
        except Exception as e:
            logging.error(f"Error initializing Claude AI: {e}")

    logging.error("No valid API keys found")
    print("\n❌ No valid API keys found! Please set GEMINI_API_KEY or ANTHROPIC_API_KEY in .env")
    return None

def create_book_search_agent():
    """Create and configure the book search agent with tools and prompts"""
    llm = initialize_llm()
    if not llm:
        return None, None

    parser = PydanticOutputParser(pydantic_object=BookResponse)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a specialized book search assistant that helps users find book information.

            Your responsibilities:
            1. Search for books based on the user's query
            2. Return ONLY the first/most relevant book found
            3. Extract exactly these fields: title, author, first year of publishing
            4. Use reliable sources for book information
            5. If multiple authors, list the primary/first author
            6. For the year, find the original first publication year, not reprints

            Available tools:
            - book_search: Search for book information using web sources

            Always structure your final response according to the format instructions below.
            Focus on accuracy and return only ONE book result.

            {format_instructions}""",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]).partial(format_instructions=parser.get_format_instructions())

    tools = [book_search_tool]

    agent = create_tool_calling_agent(
        llm=llm,
        prompt=prompt,
        tools=tools
    )

    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3
    )

    return agent_executor, parser

# Global variables for agent
agent_executor, parser = create_book_search_agent()

def save_to_csv(response):
    """Save structured book response to data.csv with proper quoting"""
    file_exists = os.path.isfile('data.csv') and os.path.getsize('data.csv') > 0
    with open('data.csv', mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_MINIMAL)
        if not file_exists:
            writer.writerow(['search_query', 'title', 'author', 'first_year_published'])
        writer.writerow([
            response.search_query,
            response.title,
            response.author,
            response.first_year_published
        ])

@app.route('/')
def serve_html():
    """Serve the main HTML interface"""
    logging.debug("Serving index.html")
    return send_file('index.html')

@app.route('/data.csv')
def serve_csv():
    """Serve the CSV data file for frontend consumption"""
    logging.debug("Serving data.csv")
    if os.path.exists('data.csv'):
        return send_file('data.csv')
    logging.warning("data.csv not found")
    return jsonify({"error": "CSV file not found"}), 404

@app.route('/search-book', methods=['POST'])
def search_book():
    """Execute book search using the AI agent - handles single or multiple books"""
    if not agent_executor:
        logging.error("Book search agent not initialized")
        return jsonify({"error": "Book search agent not initialized. Please check your API keys."}), 500

    data = request.json
    query = data.get('query')
    if not query:
        logging.warning("No query provided in request")
        return jsonify({"error": "No query provided"}), 400

    # Handle multiple books separated by newlines
    queries = [q.strip() for q in query.split('\n') if q.strip()]
    
    results = []
    for book_query in queries:
        try:
            logging.debug(f"Starting book search for query: {book_query}")
            raw_response = agent_executor.invoke({"query": f"Find information about the book: {book_query}"})
            logging.debug(f"Agent response: {raw_response}")

            if 'output' in raw_response and raw_response['output']:
                try:
                    structured_response = parser.parse(raw_response['output'])
                    structured_response.search_query = book_query  # Ensure correct query is saved
                    save_to_csv(structured_response)
                    logging.info(f"Book search completed for: {structured_response.title}")
                    results.append({
                        "title": structured_response.title,
                        "author": structured_response.author,
                        "first_year_published": structured_response.first_year_published,
                        "search_query": structured_response.search_query
                    })
                except Exception as parse_error:
                    logging.error(f"Error parsing response: {parse_error}")
                    results.append({
                        "error": f"Failed to parse: {book_query}",
                        "search_query": book_query
                    })
            else:
                logging.error("No structured output received from agent")
                results.append({
                    "error": "No results found",
                    "search_query": book_query
                })
        except Exception as e:
            logging.error(f"Book search failed for {book_query}: {e}")
            results.append({
                "error": f"Search failed: {str(e)}",
                "search_query": book_query
            })
    
    return jsonify({
        "status": f"Book search completed for {len(queries)} book(s)",
        "results": results,
        "total_searched": len(queries)
    })

@app.route('/clear', methods=['POST'])
def clear_all():
    """Clear all research data"""
    try:
        if os.path.exists('data.csv'):
            os.remove('data.csv')
            logging.info("data.csv cleared")
        return jsonify({"status": "All research data cleared successfully"})
    except Exception as e:
        logging.error(f"Error clearing data: {e}")
        return jsonify({"error": f"Failed to clear data: {str(e)}"}), 500

@app.route('/analyze-file', methods=['POST'])
def analyze_file():
    """Analyze uploaded text file for book information"""
    if not agent_executor:
        logging.error("Book search agent not initialized")
        return jsonify({"error": "Book search agent not initialized. Please check your API keys."}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
        
    if not file.filename.endswith('.txt'):
        return jsonify({"error": "Only .txt files are supported"}), 400

    try:
        # Save file temporarily
        filename = secure_filename(file.filename)
        temp_path = os.path.join('/tmp', filename)
        file.save(temp_path)
        
        # Read and analyze the file content
        with open(temp_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Clean up the temporary file immediately
        os.remove(temp_path)
        
        # Extract potential book information from content
        logging.debug(f"Analyzing file content: {content[:200]}...")
        raw_response = agent_executor.invoke({
            "query": f"Analyze this text and extract all book information (title, author, year). Find multiple books if present: {content[:3000]}"
        })
        
        if 'output' in raw_response and raw_response['output']:
            try:
                structured_response = parser.parse(raw_response['output'])
                structured_response.search_query = f"File: {filename}"
                save_to_csv(structured_response)
                logging.info(f"File analysis completed: {structured_response.title}")
                return jsonify({
                    "status": "File analysis completed successfully",
                    "filename": filename,
                    "title": structured_response.title,
                    "author": structured_response.author,
                    "first_year_published": structured_response.first_year_published,
                    "search_query": structured_response.search_query
                })
            except Exception as parse_error:
                logging.error(f"Error parsing file analysis response: {parse_error}")
                return jsonify({"error": f"Failed to parse file analysis results: {str(parse_error)}"}), 500
        else:
            logging.error("No structured output from file analysis")
            return jsonify({"error": "No book information found in file"}), 500
            
    except Exception as e:
        # Clean up file if it still exists
        if os.path.exists(temp_path):
            os.remove(temp_path)
        logging.error(f"File analysis failed: {e}")
        return jsonify({"error": f"File analysis failed: {str(e)}"}), 500

@app.route('/clear/<query>', methods=['POST'])
def clear_search(query):
    """Clear specific book search"""
    if not os.path.exists('data.csv'):
        logging.warning("data.csv not found for clearing search")
        return jsonify({"status": "No data to clear"})

    try:
        rows = []
        with open('data.csv', mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            header = next(reader, None)
            for row in reader:
                if row and row[0] != query:
                    rows.append(row)

        with open('data.csv', mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, quoting=csv.QUOTE_MINIMAL)
            if header:
                writer.writerow(header)
            writer.writerows(rows)

        logging.info(f"Search '{query}' cleared")
        return jsonify({"status": f"Search '{query}' cleared successfully"})
    except Exception as e:
        logging.error(f"Error clearing search {query}: {e}")
        return jsonify({"error": f"Failed to clear search: {str(e)}"}), 500

if __name__ == '__main__':
    if not agent_executor:
        print("\n⚠️ Warning: Book search agent could not be initialized!")
        print("Please ensure you have set GEMINI_API_KEY or ANTHROPIC_API_KEY in .env")
        print("The web interface will still load, but book search functionality will be unavailable.")

    logging.info("Starting Flask server")
    app.run(host='0.0.0.0', port=5000, debug=True)