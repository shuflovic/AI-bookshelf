import os
import csv
import logging
from flask import Flask, send_file, jsonify, request
from dotenv import load_dotenv
import time

# Configure logging for debugging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-for-research-assistant")

# Load environment variables
load_dotenv()

def create_demo_response(query):
    """Create a demo response for testing without API keys"""
    return {
        "topic": query,
        "summary": f"This is a demo response for '{query}'. In the real version with API keys, the AI would search the web and Wikipedia to provide comprehensive research with current information, facts, and reliable sources.",
        "sources": [
            "https://example.com/demo-source-1",
            "https://example.com/demo-source-2",
            "Wikipedia: Demo Article"
        ],
        "tools_used": [
            "DuckDuckGo Search (Demo)",
            "Wikipedia Search (Demo)",
            "File Save (Demo)"
        ]
    }

def save_to_csv(response):
    """Save structured research response to data.csv"""
    file_exists = os.path.isfile('data.csv') and os.path.getsize('data.csv') > 0
    with open('data.csv', mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['topic', 'summary', 'sources', 'tools_used'])
        writer.writerow([
            response["topic"],
            response["summary"],
            ';'.join(response["sources"]),  # Join sources with semicolon
            ';'.join(response["tools_used"])  # Join tools with semicolon
        ])

@app.route('/')
def serve_html():
    """Serve the main HTML interface"""
    return send_file('index.html')

@app.route('/data.csv')
def serve_csv():
    """Serve the CSV data file for frontend consumption"""
    if os.path.exists('data.csv'):
        return send_file('data.csv')
    return jsonify({"error": "CSV file not found"}), 404

@app.route('/research', methods=['POST'])
def run_research():
    """Execute demo research"""
    query = request.json.get('query')
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    try:
        logging.debug(f"Starting demo research for query: {query}")
        
        # Simulate processing time
        time.sleep(1)
        
        # Create demo response
        response = create_demo_response(query)
        save_to_csv(response)
        
        return jsonify({
            "status": "Demo research completed (Add GEMINI_API_KEY for real AI research)",
            "topic": response["topic"],
            "summary": response["summary"],
            "sources": response["sources"],
            "tools_used": response["tools_used"]
        })
        
    except Exception as e:
        logging.error(f"Demo research failed: {e}")
        return jsonify({"error": f"Demo research failed: {str(e)}"}), 500

@app.route('/clear', methods=['POST'])
def clear_all():
    """Clear all research data"""
    try:
        if os.path.exists('data.csv'):
            os.remove('data.csv')
        return jsonify({"status": "All research data cleared successfully"})
    except Exception as e:
        logging.error(f"Error clearing data: {e}")
        return jsonify({"error": f"Failed to clear data: {str(e)}"}), 500

@app.route('/clear/<topic>', methods=['POST'])
def clear_topic(topic):
    """Clear specific research topic"""
    if not os.path.exists('data.csv'):
        return jsonify({"status": "No data to clear"})
    
    try:
        # Read existing data
        rows = []
        with open('data.csv', mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            header = next(reader, None)
            for row in reader:
                if row and row[0] != topic:
                    rows.append(row)
        
        # Rewrite CSV without the specified topic
        with open('data.csv', mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            if header:
                writer.writerow(header)
            writer.writerows(rows)
        
        return jsonify({"status": f"Topic '{topic}' cleared successfully"})
    except Exception as e:
        logging.error(f"Error clearing topic {topic}: {e}")
        return jsonify({"error": f"Failed to clear topic: {str(e)}"}), 500

if __name__ == '__main__':
    print("\n🚀 Demo Mode: Research Assistant")
    print("💡 This is running in demo mode with simulated responses")
    print("🔑 Add GEMINI_API_KEY to enable real AI research")
    print("🌐 Web interface: http://localhost:5000")
    
    app.run(host='0.0.0.0', port=5000, debug=True)