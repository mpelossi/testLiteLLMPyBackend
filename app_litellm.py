"""
Backend using LiteLLM Python SDK
- Native LiteLLM library
- Handles provider-specific details automatically
- Simple unified interface
"""
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from flask_cors import CORS
import os
import json
from dotenv import load_dotenv
import litellm

load_dotenv()

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# Configuration
LITELLM_API_KEY = os.environ.get("LITELLM_API_KEY", "")
LITELLM_BASE_URL = "https://litellm.sph-prod.ethz.ch"


@app.route("/", methods=["GET"])
def serve_frontend():
    return send_from_directory('.', 'index.html')


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "backend": "litellm-sdk"})


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        if not data or "message" not in data:
            return jsonify({"error": "Missing 'message' in request body"}), 400
        
        user_message = data["message"]
        conversation_history = data.get("conversation_history", [])
        
        messages = conversation_history.copy()
        messages.append({"role": "user", "content": user_message})
        
        ###########################################
        # LiteLLM SDK - Simple unified interface
        # Use openai/ prefix to route through the proxy
        # The proxy will handle the actual model routing
        ###########################################
        response = litellm.completion(
            model="gemini/gemini-flash-lite-latest",
            messages=messages,
            api_key=LITELLM_API_KEY,
            api_base=LITELLM_BASE_URL
        )
        ###########################################
        
        return jsonify({
            "response": response.choices[0].message.content,
            "model": response.model,
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
    try:
        data = request.get_json()
        if not data or "message" not in data:
            return jsonify({"error": "Missing 'message' in request body"}), 400
        
        messages = data.get("conversation_history", []).copy()
        messages.append({"role": "user", "content": data["message"]})
        
        def generate():
            ###########################################
            # LiteLLM SDK - Streaming
            # Use openai/ prefix to route through the proxy
            ###########################################
            response = litellm.completion(
                model="gemini/gemini-flash-lite-latest",
                messages=messages,
                api_key=LITELLM_API_KEY,
                api_base=LITELLM_BASE_URL,
                stream=True
            )
            
            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield f"data: {json.dumps({'content': chunk.choices[0].delta.content})}\n\n"
            ###########################################
            
            yield "data: [DONE]\n\n"
        
        return Response(stream_with_context(generate()), mimetype='text/event-stream')
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/models", methods=["GET"])
def list_models():
    try:
        import requests
        response = requests.get(
            f"{LITELLM_BASE_URL}/v1/models",
            headers={"Authorization": f"Bearer {LITELLM_API_KEY}"}
        )
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({"error": response.text}), response.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/structured", methods=["POST"])
def structured_output():
    """
    Structured output endpoint using JSON mode.
    Forces the model to return valid JSON.
    """
    try:
        data = request.get_json()
        if not data or "prompt" not in data:
            return jsonify({"error": "Missing 'prompt' in request body"}), 400
        
        user_prompt = data["prompt"]
        json_schema = data.get("schema", None)
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Always respond with valid JSON only."},
            {"role": "user", "content": user_prompt}
        ]
        
        ###########################################
        # LiteLLM SDK - Structured Output (JSON Mode)
        ###########################################
        completion_kwargs = {
            "model": "openai/gemini-flash-lite-latest",
            "messages": messages,
            "api_key": LITELLM_API_KEY,
            "api_base": LITELLM_BASE_URL,
        }
        
        if json_schema:
            # Use json_schema format for strict validation
            completion_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": json_schema
                }
            }
        else:
            # Use basic json_object mode
            completion_kwargs["response_format"] = {"type": "json_object"}
        
        response = litellm.completion(**completion_kwargs)
        ###########################################
        
        raw_response = response.choices[0].message.content
        
        # Parse the JSON response
        try:
            parsed = json.loads(raw_response)
            return jsonify({
                "success": True,
                "data": parsed,
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
    print("=" * 50)
    print("BACKEND: LiteLLM SDK")
    print("=" * 50)
    print(f"API Key: {'Yes' if LITELLM_API_KEY else 'NO'}")
    print(f"Base URL: {LITELLM_BASE_URL}")
    print("=" * 50)
    
    app.run(host="0.0.0.0", port=5000, debug=True)
