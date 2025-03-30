import requests
import os
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # This enables CORS for all routes

API_KEY = os.getenv("MISTRAL_API_KEY")

if not API_KEY:
    raise ValueError("MISTRAL_API_KEY is not set. Please configure it in the environment variables.")


API_URL = "https://api.mistral.ai/v1/chat/completions"

def load_cv_content():
    cv_text = """
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
    return cv_text

def ask_mistral(question, cv_text):
    """Send a prompt to Mistral AI and return the response."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    
    # Improved prompt design with better instructions
    system_prompt = """You are a helpful assistant for a CV chatbot. 
    - For simple greetings, respond briefly and naturally.
    - Only mention CV information when directly asked about it.
    - Keep responses concise and relevant to the question.
    - Don't list everything from the CV unless specifically requested.
    - Be conversational but professional.
    - If asked about skills, experience, or other specific CV information, provide only the relevant details."""
    
    # Create a more contextual prompt based on the question type
    user_prompt = f"CV Information:\n{cv_text}\n\nUser: {question}"
    
    data = {
        "model": "mistral-small-latest",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.4,  # Lower temperature for more consistent responses
        "max_tokens": 500    
    }
    
    try:
        response = requests.post(API_URL, json=data, headers=headers)
        response.raise_for_status()
        
        return response.json().get("choices", [{}])[0].get("message", {}).get("content", "No response.")
    except requests.exceptions.HTTPError as e:
        return f"Error: {response.status_code}, {response.text}"
    except requests.exceptions.RequestException as e:
        return f"Request error: {e}"
    except Exception as e:
        return f"Unexpected error: {e}"

@app.route('/', methods=['POST'])
def chat():
    """Chatbot endpoint for CV-related questions."""
    try:
        cv_text = load_cv_content()
        data = request.json
        question = data.get('message', '').strip()  # Add strip() to remove whitespace
        
        if not question:
            return jsonify({'error': 'No message provided'}), 400
            
        # Improved greeting detection
        greeting_words = ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening"]
        is_greeting = any(word in question.lower().split() for word in greeting_words)
        
        if is_greeting:
            response = "Hi there! I can help you find information from Seyed's CV. What would you like to know?"
        else:
            # Add basic input validation
            if len(question) > 500:  # Prevent extremely long questions
                return jsonify({'error': 'Question too long. Please keep it under 500 characters.'}), 400
                
            response = ask_mistral(question, cv_text)
            
        return jsonify({'response': response})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500



if __name__ == '__main__':
    app.run(debug=False)
