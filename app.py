import os
import csv
import logging
from flask import Flask, send_file, jsonify, request
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool, website_tool

# Configure logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-for-research-assistant")

# Load environment variables
load_dotenv()

class ResearchResponse(BaseModel):
    """Structured response model for research output"""
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

def initialize_llm():
    """Initialize the AI model with support for multiple providers"""
    google_key = os.getenv('GEMINI_API_KEY')
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')

    if google_key:
        try:
            logging.info("Initializing Google Gemini")
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                api_key=google_key,
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
                api_key=anthropic_key
            )
            return llm
        except Exception as e:
            logging.error(f"Error initializing Claude AI: {e}")

    logging.error("No valid API keys found")
    print("\n❌ No valid API keys found! Please set GEMINI_API_KEY or ANTHROPIC_API_KEY in .env")
    return None

def create_research_agent():
    """Create and configure the research agent with tools and prompts"""
    llm = initialize_llm()
    if not llm:
        return None, None

    parser = PydanticOutputParser(pydantic_object=ResearchResponse)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert research assistant that helps users gather comprehensive information on any topic.

            Your responsibilities:
            1. Use available tools to search for current and accurate information
            2. Provide detailed summaries with key insights (at least 200 words)
            3. Always cite your sources and mention which tools you used
            4. Focus on factual, reliable information from credible sources
            5. If a specific website URL is provided, prioritize fetching content from that site using the website tool

            Available tools:
            - search: Use DuckDuckGo to find current web information
            - wikipedia: Query Wikipedia for encyclopedic knowledge
            - website: Fetch and summarize content from a specific website URL
            - save_text_to_file: Save research results to data.csv

            Always structure your final response according to the format instructions below.
            Be thorough, concise, and ensure all information is accurate and well-sourced.

            {format_instructions}""",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]).partial(format_instructions=parser.get_format_instructions())

    tools = [search_tool, wiki_tool, website_tool, save_tool]

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
        max_iterations=5
    )

    return agent_executor, parser

# Global variables for agent
agent_executor, parser = create_research_agent()

def save_to_csv(response):
    """Save structured research response to data.csv with proper quoting"""
    file_exists = os.path.isfile('data.csv') and os.path.getsize('data.csv') > 0
    with open('data.csv', mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_MINIMAL)
        if not file_exists:
            writer.writerow(['topic', 'summary', 'sources', 'tools_used'])
        writer.writerow([
            response.topic,
            response.summary,
            ';'.join(response.sources),
            ';'.join(response.tools_used)
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

@app.route('/research', methods=['POST'])
def run_research():
    """Execute research using the AI agent"""
    if not agent_executor:
        logging.error("Research agent not initialized")
        return jsonify({"error": "Research agent not initialized. Please check your API keys."}), 500

    data = request.json
    query = data.get('query')
    website = data.get('website', '')  # Optional website URL
    if not query:
        logging.warning("No query provided in request")
        return jsonify({"error": "No query provided"}), 400

    # Construct query with website if provided
    full_query = f"{query} (use website tool for {website})" if website else query

    try:
        logging.debug(f"Starting research for query: {full_query}")
        raw_response = agent_executor.invoke({"query": full_query})
        logging.debug(f"Agent response: {raw_response}")

        if 'output' in raw_response and raw_response['output']:
            try:
                structured_response = parser.parse(raw_response['output'])
                save_to_csv(structured_response)
                logging.info(f"Research completed for topic: {structured_response.topic}")
                return jsonify({
                    "status": "Research completed successfully",
                    "topic": structured_response.topic,
                    "summary": structured_response.summary,
                    "sources": structured_response.sources,
                    "tools_used": structured_response.tools_used
                })
            except Exception as parse_error:
                logging.error(f"Error parsing response: {parse_error}")
                return jsonify({"error": f"Failed to parse research results: {str(parse_error)}"}), 500
        else:
            logging.error("No structured output received from agent")
            return jsonify({"error": "No structured output received from research agent"}), 500
    except Exception as e:
        logging.error(f"Research failed: {e}")
        return jsonify({"error": f"Research failed: {str(e)}"}), 500

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

@app.route('/clear/<topic>', methods=['POST'])
def clear_topic(topic):
    """Clear specific research topic"""
    if not os.path.exists('data.csv'):
        logging.warning("data.csv not found for clearing topic")
        return jsonify({"status": "No data to clear"})

    try:
        rows = []
        with open('data.csv', mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            header = next(reader, None)
            for row in reader:
                if row and row[0] != topic:
                    rows.append(row)

        with open('data.csv', mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, quoting=csv.QUOTE_MINIMAL)
            if header:
                writer.writerow(header)
            writer.writerows(rows)

        logging.info(f"Topic '{topic}' cleared")
        return jsonify({"status": f"Topic '{topic}' cleared successfully"})
    except Exception as e:
        logging.error(f"Error clearing topic {topic}: {e}")
        return jsonify({"error": f"Failed to clear topic: {str(e)}"}), 500

if __name__ == '__main__':
    if not agent_executor:
        print("\n⚠️ Warning: Research agent could not be initialized!")
        print("Please ensure you have set GEMINI_API_KEY or ANTHROPIC_API_KEY in .env")
        print("The web interface will still load, but research functionality will be unavailable.")

    logging.info("Starting Flask server")
    app.run(host='0.0.0.0', port=5000, debug=True)