import asyncio
import json
import logging
from flask import Blueprint, jsonify, request, Response
from flask_cors import CORS

# Assuming orchestrator and other necessary components are accessible
# from a global app context or passed during blueprint registration
# For now, we'll use a placeholder for the orchestrator
# from ..agents.orchestrator import AgentOrchestrator

# Placeholder for the orchestrator instance
# In a real Flask app, this would be initialized once and passed around
# For now, we'll assume it's globally accessible or mocked.
class MockOrchestrator:
    def __init__(self):
        self.logger = logging.getLogger("mock_orchestrator")
        self.mock_survey_state = {
            "url": "https://example.com/survey/123",
            "current_question": {
                "id": "q1",
                "text": "What is your favorite color?",
                "type": "multiple_choice",
                "options": ["Red", "Blue", "Green", "Yellow"],
                "element_id": "question_div_1"
            },
            "agent_decision": {
                "agent_id": "responder-1",
                "selected_answer": "Blue",
                "confidence": 0.85,
                "timestamp": "2023-10-27T10:00:00Z"
            },
            "browser_screenshot_base64": "", # Placeholder for base64 image
            "status": "waiting_for_input",
            "log_messages": []
        }

    async def get_live_survey_state(self) -> Dict[str, Any]:
        # In a real scenario, this would fetch the actual live state from the Navigator Agent
        self.logger.info("Fetching mock live survey state")
        return self.mock_survey_state

    async def send_manual_override(self, override_data: Dict[str, Any]) -> bool:
        self.logger.info(f"Received manual override: {override_data}")
        # Simulate applying override
        if "selected_answer" in override_data:
            self.mock_survey_state["current_question"]["selected_answer"] = override_data["selected_answer"]
            self.mock_survey_state["agent_decision"]["selected_answer"] = override_data["selected_answer"]
            self.mock_survey_state["status"] = "manual_override_applied"
        if "action" in override_data:
            if override_data["action"] == "pause":
                self.mock_survey_state["status"] = "paused"
            elif override_data["action"] == "resume":
                self.mock_survey_state["status"] = "running"
            elif override_data["action"] == "skip":
                self.mock_survey_state["status"] = "question_skipped"
            elif override_data["action"] == "submit_manually":
                self.mock_survey_state["status"] = "manual_submission"
        return True

    async def get_agent_logs(self, limit: int = 10) -> List[Dict[str, Any]]:
        # In a real scenario, this would fetch logs from the Observer Agent
        return self.mock_survey_state["log_messages"][-limit:]

    async def get_login_status(self) -> Dict[str, Any]:
        return {"status": "logged_in", "platform": "Swagbucks", "last_login": "2023-10-27T09:55:00Z"}

mock_orchestrator = MockOrchestrator()


live_watch_bp = Blueprint('live_watch', __name__)
CORS(live_watch_bp) # Enable CORS for this blueprint

logger = logging.getLogger(__name__)

@live_watch_bp.route('/live_state', methods=['GET'])
async def get_live_state():
    """Endpoint to stream live survey state updates."""
    async def generate_updates():
        while True:
            # In a real app, this would get updates from the orchestrator
            # For now, we'll just send the mock state periodically
            state = await mock_orchestrator.get_live_survey_state()
            yield f"data: {json.dumps(state)}\n\n"
            await asyncio.sleep(1) # Send updates every second

    return Response(generate_updates(), mimetype='text/event-stream')

@live_watch_bp.route('/manual_override', methods=['POST'])
async def manual_override():
    """Endpoint to receive manual override commands."""
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    logger.info(f"Received manual override request: {data}")
    success = await mock_orchestrator.send_manual_override(data)
    
    if success:
        return jsonify({"status": "success", "message": "Override applied"}), 200
    else:
        return jsonify({"status": "error", "message": "Failed to apply override"}), 500

@live_watch_bp.route('/agent_logs', methods=['GET'])
async def get_agent_logs_endpoint():
    """Endpoint to get agent logs."""
    limit = request.args.get('limit', type=int, default=10)
    logs = await mock_orchestrator.get_agent_logs(limit)
    return jsonify(logs)

@live_watch_bp.route('/login_status', methods=['GET'])
async def get_login_status_endpoint():
    """Endpoint to get login status."""
    status = await mock_orchestrator.get_login_status()
    return jsonify(status)

