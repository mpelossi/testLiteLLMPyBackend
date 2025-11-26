# LiteLLM Proxy Flask Backend

A Flask backend for interacting with the LiteLLM proxy at ETH Zurich, with a web frontend for testing.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment:**
   Create a `.env` file with your API key:
   ```
   LITELLM_API_KEY=your-api-key-here
   ```

3. **Run the server:**
   ```bash
   python app_openai.py
   ```

4. **Open the frontend:**
   Navigate to `http://localhost:5000` in your browser.

## Project Structure

| File | Description |
|------|-------------|
| `app_openai.py` | ✅ **Working** - Flask backend using OpenAI Python SDK |
| `app_litellm.py` | ⚠️ **Does not work** - Flask backend using LiteLLM SDK (kept for reference) |
| `index.html` | Frontend with chat interface and test buttons |
| `requirements.txt` | Python dependencies |
| `.env` | Environment variables (API key) |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serves the frontend |
| `/health` | GET | Health check |
| `/models` | GET | List available models |
| `/chat` | POST | Chat completion |
| `/structured` | POST | Structured output (JSON mode) |

## Usage

### Chat
Send a message and get a response:
```json
POST /chat
{
  "message": "Hello, how are you?"
}
```

### Structured Output
Get JSON-formatted responses:
```json
POST /structured
{
  "prompt": "List 3 countries with their capitals",
  "schema": { ... }  // optional JSON schema
}
```

## Configuration

- **LiteLLM Proxy:** `https://litellm.sph-prod.ethz.ch`
- **Model:** `gemini/gemini-flash-lite-latest`

## Note on SDK Alternatives

This project includes two backend implementations for comparison:

1. **`app_openai.py`** (recommended) - Uses the OpenAI Python SDK with a custom `base_url` pointing to the LiteLLM proxy. This approach works reliably.

2. **`app_litellm.py`** - Uses the LiteLLM Python SDK directly. **This does not work** due to issues with provider prefix routing through the proxy. Kept for reference only.
