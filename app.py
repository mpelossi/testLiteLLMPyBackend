from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from flask_cors import CORS
import os
import json
from dotenv import load_dotenv
import litellm

load_dotenv()

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# LiteLLM configuration
LITELLM_BASE_URL = "https://litellm.sph-prod.ethz.ch"
LITELLM_API_KEY = os.environ.get("LITELLM_API_KEY", "")

@app.route("/", methods=["GET"])
def serve_frontend():
    """Serve the frontend HTML page."""
    return send_from_directory('.', 'index.html')


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "message": "Backend is running"})


@app.route("/chat", methods=["POST"])
def chat():
    """
    Chat endpoint that forwards requests to LiteLLM API using gemini-flash-lite-latest model.
    """
    try:
        data = request.get_json()
        
        if not data or "message" not in data:
            return jsonify({"error": "Missing 'message' in request body"}), 400
        
        user_message = data["message"]
        conversation_history = data.get("conversation_history", [])
        
        # Build messages array
        messages = conversation_history.copy()
        messages.append({"role": "user", "content": user_message})
        
        # Call LiteLLM SDK
        response = litellm.completion(
            model="gemini/gemini-flash-lite-latest",
            messages=messages,
            api_key=LITELLM_API_KEY,
            api_base=LITELLM_BASE_URL
        )
        
        assistant_message = response.choices[0].message.content
        
        return jsonify({
            "response": assistant_message,
            "model": "gemini-flash-lite-latest",
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        })
        
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/chat/stream", methods=["POST"])
def chat_stream():
    """
    Streaming chat endpoint for real-time responses.
    """
    try:
        data = request.get_json()
        
        if not data or "message" not in data:
            return jsonify({"error": "Missing 'message' in request body"}), 400
        
        user_message = data["message"]
        conversation_history = data.get("conversation_history", [])
        
        messages = conversation_history.copy()
        messages.append({"role": "user", "content": user_message})
        
        def generate():
            response = litellm.completion(
                model="gemini-flash-lite-latest",
                messages=messages,
                api_key=LITELLM_API_KEY,
                api_base=LITELLM_BASE_URL,
                stream=True
            )
            
            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield f"data: {json.dumps({'content': chunk.choices[0].delta.content})}\n\n"
            
            yield "data: [DONE]\n\n"
        
        return Response(stream_with_context(generate()), mimetype='text/event-stream')
        
    except Exception as e:
        print(f"Stream error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/models", methods=["GET"])
def list_models():
    """
    List available models from LiteLLM.
    """
    try:
        # Return commonly available models
        models = {
            "data": [
                {"id": "gemini-flash-lite-latest", "object": "model"},
                {"id": "gpt-4", "object": "model"},
                {"id": "gpt-3.5-turbo", "object": "model"},
            ]
        }
        return jsonify(models)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/structured", methods=["POST"])
def structured_output():
    """
    Endpoint for structured data extraction with JSON output.
    """
    try:
        data = request.get_json()
        
        if not data or "prompt" not in data:
            return jsonify({"error": "Missing 'prompt' in request body"}), 400
        
        user_prompt = data["prompt"]
        input_data = data.get("data", "")
        schema = data.get("schema", None)
        
        system_message = """You are a helpful assistant that extracts and returns structured data.
Always respond with valid JSON only. No explanations, no markdown, just pure JSON."""
        
        full_prompt = user_prompt
        if input_data:
            full_prompt += f"\n\nData to process:\n{input_data}"
        if schema:
            full_prompt += f"\n\nExpected JSON schema:\n{json.dumps(schema, indent=2)}"
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": full_prompt}
        ]
        
        response = litellm.completion(
            model="gemini-flash-lite-latest",
            messages=messages,
            api_key=LITELLM_API_KEY,
            api_base=LITELLM_BASE_URL,
            temperature=0.1
        )
        
        raw_response = response.choices[0].message.content
        
        # Try to parse as JSON
        try:
            cleaned = raw_response.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            
            parsed_json = json.loads(cleaned.strip())
            return jsonify({
                "success": True,
                "data": parsed_json,
                "raw_response": raw_response,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            })
        except json.JSONDecodeError:
            return jsonify({
                "success": False,
                "error": "Failed to parse response as JSON",
                "raw_response": raw_response
            })
        
    except Exception as e:
        print(f"Structured error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print(f"API Key loaded: {'Yes (' + LITELLM_API_KEY[:10] + '...)' if LITELLM_API_KEY else 'NO - Check .env file!'}")
    print(f"LiteLLM URL: {LITELLM_BASE_URL}")
    
    if not LITELLM_API_KEY:
        print("Warning: LITELLM_API_KEY environment variable not set!")
        print("Make sure your .env file contains: LITELLM_API_KEY=your-key")
    
    app.run(host="0.0.0.0", port=5000, debug=True)
