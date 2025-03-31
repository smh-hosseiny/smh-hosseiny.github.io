import requests
import os
import logging
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # This enables CORS for all routes

API_KEY = os.getenv("MISTRAL_API_KEY")

if not API_KEY:
    logger.error("MISTRAL_API_KEY environment variable is not set")
    raise ValueError("MISTRAL_API_KEY is not set. Please configure it in the environment variables.")

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["100 per day", "10 per minute"]
)

API_URL = "https://api.mistral.ai/v1/chat/completions"

# Add near the top with other constants
SYSTEM_PROMPT = """You are a helpful assistant for a CV chatbot. 
- For simple greetings, respond briefly and naturally.
- Only mention CV information when directly asked about it.
- Keep responses concise and relevant to the question.
- Don't list everything from the CV unless specifically requested.
- Be conversational but professional.
- If asked about skills, experience, or other specific CV information, provide only the relevant details."""

# Add near the top
cv_content = None

def load_cv_content():
    """Load CV content from a file with caching"""
    global cv_content
    if cv_content is not None:
        return cv_content
        
    try:
        with open('cv.txt', 'r') as file:
            cv_content = file.read()
            return cv_content
    except FileNotFoundError:
        logger.error("CV content file not found")
        # Fallback to hardcoded content
        cv_content = """
            I am a graduate researcher in Electrical and Computer Engineering at York University, 
            specializing in monocular metric depth estimation at the Elder Lab under the supervision 
            of Prof. James Elder.
            Prior to this, I completed my B.Sc. in Electrical Engineering at the University of Tehran (2021), 
            where I conducted research on 3D reconstruction of symmetrical objects from single images at the 
            Computational Audio-Vision Lab with Prof. Reshad Hosseini.
            Research Interests:
            - Computer Vision
            - 3D Scene Understanding
            - Medical Image Analysis
            Publications:
            - The third monocular depth estimation challenge. Spencer, et al., CVPR 2024.
            - CTtrack: A CNN+Transformer-based framework for fiber orientation estimation & tractography. 
            S.M.H. Hosseini, et al., Neuroscience Informatics, 2022.
            - Single-view 3D Reconstruction of Surface of Revolution. S.M.H. Hosseini, S.M. Nasiri, 
            R. Hosseini, H. Moradi, 2022
            Skills:
            - Programming: Python, C, C++, MATLAB, SQL, Bash
            - Frameworks: TensorFlow, Keras, PyTorch, OpenCV
            - Tools: Git, Docker, AWS
            Current Position:
            - Computer Vision Researcher at Elder Lab, York University (2024–Present, Toronto, ON)
                """
        return cv_content

def ask_mistral(question, cv_text):
    """Send a prompt to Mistral AI and return the response."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    
    # Create a more contextual prompt based on the question type
    user_prompt = f"CV Information:\n{cv_text}\n\nUser: {question}"
    
    data = {
        "model": "mistral-small-latest",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.4,  # Lower temperature for more consistent responses
        "max_tokens": 500    
    }
    
    try:
        response = requests.post(API_URL, json=data, headers=headers, timeout=10)
        response.raise_for_status()
        
        return response.json().get("choices", [{}])[0].get("message", {}).get("content", "No response.")
    except requests.exceptions.Timeout:
        logger.error("Request to Mistral API timed out")
        return "Sorry, the request timed out. Please try again."
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP Error: {e.response.status_code} - {e.response.text}")
        return f"Error: {response.status_code}, {response.text}"
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {str(e)}")
        return f"Request error: {e}"
    except Exception as e:
        logger.error(f"Unexpected error in ask_mistral: {str(e)}", exc_info=True)
        return f"Unexpected error: {e}"

def validate_mistral_response(response):
    """Validate the response from Mistral API"""
    if not response or response.isspace():
        return False
    if len(response) > 1000:  # Prevent extremely long responses
        return False
    return True

@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

# Add this to serve static files
@app.route('/assets/<path:path>')
def serve_static(path):
    return send_from_directory('assets', path)

@app.route('/api/chat', methods=['GET', 'POST'])
@limiter.limit("10 per minute")
def chat():
    """Chatbot endpoint for CV-related questions."""
    if request.method == 'GET':
        return jsonify({'message': 'API is running'})
        
    try:
        cv_text = load_cv_content()
        data = request.json
        question = data.get('message', '').strip()
        
        if not question:
            logger.warning("Empty message received")
            return jsonify({'error': 'No message provided'}), 400
            
        # Log incoming questions
        logger.info(f"Received question: {question}")
            
        # Improved greeting detection
        greeting_words = ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening"]
        is_greeting = any(word in question.lower().split() for word in greeting_words)
        
        if is_greeting:
            response = "Hi there! I can help you find information from Seyed's CV. What would you like to know?"
        else:
            # Add basic input validation
            if len(question) > 500:  # Prevent extremely long questions
                logger.warning(f"Question too long: {len(question)} characters")
                return jsonify({'error': 'Question too long. Please keep it under 500 characters.'}), 400
                
            response = ask_mistral(question, cv_text)
            
            if not validate_mistral_response(response):
                logger.error("Invalid response received from Mistral API")
                return jsonify({'error': 'Sorry, I could not generate a valid response. Please try again.'}), 500
            
        return jsonify({'response': response})
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        return jsonify({'error': 'An unexpected error occurred. Please try again later.'}), 500

@app.route('/chat')
def serve_chat():
    return send_from_directory('.', 'chat.html')

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
