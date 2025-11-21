import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from google.genai import types

# --- 1. FLASK APP SETUP ---
app = Flask(__name__)
# IMPORTANT: Enable CORS to allow your index.html to communicate with Flask.
CORS(app) 

# --- 2. GEMINI CLIENT INITIALIZATION ---
# 1. Import the 'os' module to access environment variables
import os 
# ... other imports like from google import genai, types, etc.
# ...

# 2. **CRITICAL FOR DEPLOYMENT**: Read the API Key from the environment variable 
#    We will set this variable (GEMINI_API_KEY) securely on the Render server later.
API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    print("FATAL ERROR: GEMINI_API_KEY environment variable is NOT set!")
    # In a real deployed app, you might raise an error here to prevent starting.
    # We will assume you set it on Render, so this only acts as a safety check.

# 3. Initialize the Gemini Client using the key
try:
    client = genai.Client(api_key=API_KEY)
except Exception as e:
    print(f"Error initializing Gemini client: {e}")
    client = None # Set client to None if initialization fails

# --- 3. HELPER FUNCTIONS ---

def convert_to_gemini_history(app_history):
    """
    Converts the app's history format (list of {role: str, text: str}) 
    into the format required by the Gemini API (list of Content objects).
    
    Uses the ultra-simplified Part constructor to avoid argument count errors.
    """
    gemini_history = []
    
    for message in app_history:
        role = message.get('role')
        text = message.get('text')
        
        # Only process 'user' and 'model' roles for the chat history/context
        if role in ['user', 'model'] and text:
            api_role = 'user' if role == 'user' else 'model'
            gemini_history.append(
                types.Content(
                    role=api_role, 
                    # ULTRA-SIMPLIFIED FIX: Use direct Part constructor
                    parts=[types.Part(text=text)] 
                )
            )
            
    return gemini_history


# --- 4. FLASK API ROUTE (CHAT HANDLER) ---

@app.route('/chat', methods=['POST'])
def chat_handler():
    # If client failed to init (e.g., missing API key), return 500
    if client is None:
        return jsonify({"error": "Gemini client not initialized. Check GEMINI_API_KEY."}), 500
        
    # 1. Get JSON data from the request
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data received. Request body is empty."}), 400
            
        # Robust check for required keys from the frontend
        if 'system_instruction' not in data or 'history' not in data:
             return jsonify({"error": "Missing required keys 'system_instruction' or 'history' in JSON payload."}), 400
             
        system_instruction = data.get('system_instruction', '')
        history_from_app = data.get('history', [])
        
    except Exception as e:
        return jsonify({"error": f"Error parsing incoming JSON data: {str(e)}"}), 400

    # 2. Extract latest query and history context
    try:
        if not history_from_app:
             return jsonify({"error": "History array is empty. Cannot process request."}), 400
             
        # The latest user message is the absolute last item in the history list.
        latest_user_query = history_from_app[-1]['text']
        
        # The rest of the history (all messages BEFORE the last one) forms the context.
        context_history = history_from_app[:-1]
        
        # Convert context messages to the required API format
        gemini_history = convert_to_gemini_history(context_history)

    except IndexError:
        return jsonify({"error": "History array is malformed. Cannot extract latest query."}), 400
    except Exception as e:
        return jsonify({"error": f"Error extracting query/history: {str(e)}"}), 400

    # 3. Configure the model and send message
    try:
        config = types.GenerateContentConfig(
            system_instruction=system_instruction
        )
        
        # Create a new chat session with the previous context/history
        chat = client.chats.create(model='gemini-2.5-flash', history=gemini_history)
        
        # Send ONLY the latest query to the chat session
        response = chat.send_message(latest_user_query, config=config)
        
        # 4. Return the response text
        return jsonify({"text": response.text})

    except Exception as e:
        print(f"Gemini API Error: {str(e)}")
        # If the API call fails (e.g., key error, rate limit)
        return jsonify({"error": f"Gemini API Processing Error: {str(e)}. Check that your API key is valid."}), 500

# --- 5. RUN FLASK APP ---
if __name__ == '__main__':
    # Setting debug=False as you saw it overriding environment variables
    app.run(host='127.0.0.1', port=5000, debug=False)