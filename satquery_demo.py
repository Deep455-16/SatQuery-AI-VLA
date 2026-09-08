"""
satquery_demo.py — SatQuery AI entry point.

Run with:
    streamlit run satquery_demo.py

Ensure your .env file contains:
    GEMINI_API_KEY=<your free key from https://aistudio.google.com/apikey>
    MONGODB_URI=<your MongoDB Atlas URI>
"""
from satquery.serve.web_server import main

if __name__ == "__main__":
    main()
