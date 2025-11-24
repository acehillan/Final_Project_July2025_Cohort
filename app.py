import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from google import genai
from google.genai import types
from google.genai.errors import APIError

# --- FLASK SETUP ---
# Create the Flask application instance.
# IMPORTANT: Explicitly defining 'template_folder' helps Gunicorn find index.html
app = Flask(__name__, template_folder='templates') 
# Enable CORS for cross-origin requests from the frontend
CORS(app) 

# --- CONFIGURATION & SECRETS ---
# Retrieve the API key from the environment variables (set on Render)
API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    print("WARNING: GEMINI_API_KEY environment variable not set. App will only function on Render.")
    
# --- GEMINI MODEL SETUP ---
def initialize_gemini_client(api_key):
    """Initializes the Gemini client."""
    if not api_key:
        return None
    try:
        # Client initialized with API key from environment
        return genai.Client(api_key=api_key)
    except Exception as e:
        print(f"Error initializing Gemini client: {e}")
        return None

client = initialize_gemini_client(API_KEY)

# --- ROUTES ---

@app.route('/')
def index():
    """Serves the main HTML page (the frontend)."""
    # NOTE: Assuming your index.html is in a 'templates' folder as per standard Flask convention
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate_content():
    """
    Handles the POST request from the frontend to generate AI content.
    Includes robust error checking and uses the 'response' field for output.
    """
    if client is None:
        return jsonify({"error": "AI client not initialized. Check server configuration."}), 503

    try:
        # 1. Get user query from the request body (Frontend now sends 'prompt')
        data = request.get_json()
        user_query = data.get('prompt', 'Please explain a topic briefly.')
        
        # 2. Define the system instruction for the Mwalimu Jua persona
        system_instruction = (
            "You are Mwalimu Jua, an encouraging, patient, and knowledgeable secondary school tutor "
            "based in Kenya. Your tone should be warm and formal. Always provide answers that are "
            "educational and suitable for a high school student, and incorporate a Kenyan or African "
            "context or example where relevant (e.g., use Kenyan currency, local landmarks, or cultural concepts). "
            "Your output must be clear, well-structured, and use Markdown for formatting."
        )

        # 3. Define the full request payload for the API
        contents = [
            {"role": "user", "parts": [{"text": user_query}]}
        ]
        
        # 4. Call the Gemini API with grounding enabled
        response = client.models.generate_content(
            model='gemini-2.5-flash-preview-09-2025',
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                tools=[{"google_search": {}}] # Enables Google Search Grounding
            ),
        )

        # 5. ROBUST CHECK: Ensure the response contains valid text before processing
        if not response.candidates or not response.text:
             print(f"Warning: Model response was blocked or empty for query: {user_query}")
             return jsonify({
                "response": "Pole sana (My apologies), Mwalimu Jua could not generate a response for that query. Please try rephrasing.",
                "sources": [] 
             }), 400 
        
        # If text is valid, proceed
        text = response.text
        
        # 6. Return the AI response using the 'response' field expected by the frontend
        return jsonify({
            "response": text, # CRITICAL: Use 'response' field
            "sources": [] 
        })

    except APIError as e:
        print(f"Final API call failed: {e}")
        return jsonify({"error": f"Gemini API Error: {str(e)}"}), 500
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return jsonify({"error": f"Server Error: {str(e)}"}), 500


# --- ENTRY POINT ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=os.environ.get('PORT', 5000))