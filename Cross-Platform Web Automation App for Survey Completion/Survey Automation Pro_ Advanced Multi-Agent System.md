# Survey Automation Pro: Advanced Multi-Agent System

This is an enhanced cross-platform web automation application designed to intelligently fill out survey questions on platforms like Swagbucks and InboxDollars. It incorporates state-of-the-art multi-agent systems, real-time monitoring, dynamic AI model integration, and robust anti-bot detection evasion techniques.

## 🚀 Key Features Implemented:

### 1. Crew-Based Role Specialization
- **Navigator Agent**: Handles web browsing, login automation, survey discovery, and platform interaction with anti-detection capabilities.
- **Extractor Agent**: Specializes in survey question parsing, metadata extraction, and question type classification.
- **Responder Agent**: Generates AI-powered responses with multi-LLM support and semantic anchoring.
- **Validator Agent**: Verifies responses, ensures quality, and provides confidence scoring.
- **Observer Agent**: Passively monitors and logs all agent activities, decisions, and inter-agent communications for debugging, benchmarking, and training.
- **Agent Orchestrator**: Central coordination system for managing the agent crew, dispatching tasks, and facilitating communication.

### 2. Survey Type Classification Layer
- **ML-Powered Classification**: Classifies surveys into categories (opinion polls, product feedback, demographic profiling, etc.) and individual questions into types (multiple choice, Likert scale, open-ended, numeric, etc.).
- **Tailored Response Strategies**: Enables specific response strategies based on survey and question types, including persona weighting and consistency requirements.

### 3. Semantic Anchoring for Question Mapping
- **Embedding-Based Similarity**: Uses SentenceTransformers for high-quality semantic embeddings to map new questions to known templates.
- **Template Management**: Manages a database of known question templates with associated response strategies.
- **Response Strategy Reuse**: Improves AI response relevance and reuses fine-tuned logic from matched templates.

### 4. Multi-LLM Backend Switching
- **Dynamic Routing**: Integrates a routing layer to dynamically select between various LLM providers (OpenAI, HuggingFace, Ollama, Groq) based on performance, cost, latency, or specific feature requirements.
- **Unified API**: Provides a consistent interface for interacting with different LLMs.
- **Health Monitoring & Analytics**: Continuously monitors backend health and tracks usage statistics.

### 5. Observer Mode for Agent Swarms
- **Comprehensive Logging**: Logs all agent decisions, interactions, and messages.
- **Behavioral Analysis Tools**: Provides methods for analyzing event logs to understand agent behavior, identify common errors, and optimize task flows.

### 6. Live Survey Watch Mode
- **Real-Time Survey Viewer**: Renders the active survey in a mirrored browser window or embedded iframe.
- **Agent Decision Overlays**: Displays agent decisions (e.g., selected answers, confidence scores) as overlays.
- **Manual Override Panel**: Allows users to pause/resume automation, edit current responses, skip questions, or submit manually.
- **Agent Confidence Meter**: Displays a confidence score per question, highlighting low-confidence responses for human review.

### 7. Multi-Agent Voting Panel and Swarm Coordination
- **Voting Mechanisms**: Implements majority vote, weighted vote (based on confidence), and confidence threshold filtering.
- **Consensus Resolution**: Orchestrator coordinates responses from multiple agents, reaching a consensus or flagging for human review if disagreement is high.

### 8. Memory Modules for Survey Context
- **Short-Term Memory**: Each agent is equipped with a `MemoryModule` to retain context across multi-page surveys.
- **Context-Aware Responses**: Enables coherent responses and better handling of follow-up questions by recalling past interactions and decisions.

### 9. Adaptive Retry Logic
- **Heuristic-Based Retries**: Implements adaptive retry logic with dynamic delays and jitter.
- **Error-Specific Configuration**: Allows custom retry strategies for different types of exceptions, enhancing robustness and efficiency.

### 10. Login Watch Mode
- **Real-Time Login Progress**: Displays login progress in real time.
- **User Intervention**: Allows users to intervene if CAPTCHA fails or platform behavior changes.

## ⚙️ Technical Architecture:

- **Backend**: Flask with CORS support for API endpoints.
- **Browser Automation**: Playwright with stealth capabilities for anti-detection.
- **AI Integration**: Multi-LLM backend switching (OpenAI, HuggingFace, Ollama, Groq) for response generation.
- **CAPTCHA Solving**: Tesseract OCR + 2captcha API.
- **Anti-Detection**: Advanced techniques including fingerprint evasion, proxy rotation, dynamic browser fingerprinting, request manipulation, and header obfuscation.
- **Database**: SQLite for task, user, and configuration management.
- **Frontend**: Modern HTML/CSS/JavaScript with Server-Sent Events (SSE) for real-time updates.
- **Agent Communication**: Asynchronous message queues for inter-agent communication.

## 🚀 Getting Started:

1.  **Navigate to the application directory:**
    `cd /home/ubuntu/survey_automation_pro/`

2.  **Install Python dependencies:**
    `pip install -r requirements.txt`

3.  **Install Playwright browsers:**
    `playwright install`

4.  **Set up environment variables:**
    Ensure you have necessary API keys (e.g., `OPENAI_API_KEY`, `HUGGINGFACE_API_KEY`, `GROQ_API_KEY`, `TWOCAPTCHA_API_KEY`) configured in your environment or in a `.env` file.

5.  **Run the application:**
    `python src/main.py`

6.  **Access the web interface:**
    Open your browser and navigate to `http://localhost:5000` for the main application, and `http://localhost:5000/live_watch.html` for the Live Survey Watch Mode.

## 🤝 Contribution:

Contributions are welcome! Please refer to the project guidelines for more information.

## 📄 License:

This project is licensed under the MIT License.
