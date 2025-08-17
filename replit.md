# Research Assistant

## Overview

This is an AI-powered research assistant application that enables users to conduct automated research on various topics. The system combines multiple AI language models (Google Gemini and Anthropic Claude) with web search capabilities and Wikipedia integration to provide comprehensive research outputs. Users can submit research queries through a web interface and receive structured summaries with sources and methodology information. The application is built with Flask as the backend web framework and uses LangChain for AI agent orchestration and tool management.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Backend Architecture
- **Web Framework**: Flask-based REST API serving as the main application server
- **AI Agent System**: LangChain framework for creating tool-calling agents that can orchestrate multiple research tasks
- **Multi-Provider LLM Support**: Fallback system supporting both Google Gemini (primary) and Anthropic Claude (fallback) language models
- **Structured Output**: Pydantic models for ensuring consistent response formatting with topic, summary, sources, and tools used

### Frontend Architecture
- **Web Interface**: HTML/CSS/JavaScript single-page application with sidebar navigation
- **Layout Design**: Two-column layout with collapsible sidebar for research history and main content area for current research
- **Responsive Design**: Mobile-friendly interface with viewport meta tags and flexible layouts

### Tool Integration System
- **Modular Tool Architecture**: Separate tools module defining individual research capabilities
- **Web Search Tool**: DuckDuckGo integration for current web information and news
- **Wikipedia Tool**: Wikipedia API integration for encyclopedic knowledge
- **Save Tool**: File system integration for persisting research results
- **Tool Input Validation**: Pydantic schemas for structured tool input validation

### Data Persistence
- **CSV Export**: Research results saved in structured CSV format for easy analysis
- **File System Storage**: Local file storage for research outputs and session data
- **Session Management**: Flask session handling with configurable secret keys

### Configuration Management
- **Environment Variables**: dotenv integration for API keys and configuration
- **Multi-Environment Support**: Separate configurations for development and production
- **Graceful Degradation**: Fallback mechanisms when optional dependencies are unavailable

## External Dependencies

### AI/ML Services
- **Google Gemini API**: Primary language model service via Google Generative AI
- **Anthropic Claude API**: Fallback language model service for advanced reasoning

### Search and Data Services
- **DuckDuckGo Search API**: Web search functionality through duckduckgo-search library
- **Wikipedia API**: Encyclopedia content access via wikipedia Python library

### Core Framework Dependencies
- **LangChain**: AI agent framework for tool orchestration and prompt management
- **Flask**: Web application framework for HTTP handling and routing
- **Pydantic**: Data validation and serialization for structured outputs

### Utility Libraries
- **python-dotenv**: Environment variable management
- **CSV**: Built-in Python library for data export functionality
- **Threading**: Concurrent processing support for research tasks
- **Logging**: Debug and error tracking capabilities