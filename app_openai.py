"""
Backend using OpenAI Python SDK
- Official OpenAI library
- Works with any OpenAI-compatible API (like LiteLLM proxy)
- Industry standard, widely used
"""
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from flask_cors import CORS
import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# Configuration
LITELLM_API_KEY = os.environ.get("LITELLM_API_KEY", "")
LITELLM_BASE_URL = "https://litellm.sph-prod.ethz.ch"

###########################################
# OpenAI SDK - Initialize client once
###########################################
client = OpenAI(
    api_key=LITELLM_API_KEY,
    base_url=LITELLM_BASE_URL
)
###########################################


@app.route("/", methods=["GET"])
def serve_frontend():
    return send_from_directory('.', 'index.html')


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "backend": "openai-sdk"})


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
        # OpenAI SDK - Standard OpenAI interface
        ###########################################
        response = client.chat.completions.create(
            model="gemini/gemini-flash-lite-latest",
            messages=messages
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
            # OpenAI SDK - Streaming
            ###########################################
            stream = client.chat.completions.create(
                model="gemini/gemini-flash-lite-latest",
                messages=messages,
                stream=True
            )
            
            for chunk in stream:
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
        ###########################################
        # OpenAI SDK - List models from API
        ###########################################
        models = client.models.list()
        return jsonify({
            "data": [{"id": m.id, "object": "model"} for m in models.data]
        })
        ###########################################
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
        # OpenAI SDK - Structured Output (JSON Mode)
        ###########################################
        if json_schema:
            # Use json_schema format for strict validation
            response = client.chat.completions.create(
                model="gemini/gemini-flash-lite-latest",
                messages=messages,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response",
                        "schema": json_schema
                    }
                }
            )
        else:
            # Use basic json_object mode
            response = client.chat.completions.create(
                model="gemini/gemini-flash-lite-latest",
                messages=messages,
                response_format={"type": "json_object"}
            )
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
    print("BACKEND: OpenAI SDK")
    print("=" * 50)
    print(f"API Key: {'Yes' if LITELLM_API_KEY else 'NO'}")
    print(f"Base URL: {LITELLM_BASE_URL}")
    print("=" * 50)
    
    app.run(host="0.0.0.0", port=5000, debug=True)
