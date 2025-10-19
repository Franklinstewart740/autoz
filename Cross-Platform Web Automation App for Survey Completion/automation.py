"""
Flask routes for survey automation API.
"""

from flask import Blueprint, jsonify, request
from flask_cors import cross_origin
import asyncio
import logging
from typing import Dict, Any

from src.automation.survey_automation import SurveyAutomation

logger = logging.getLogger(__name__)

automation_bp = Blueprint('automation', __name__)

# Global automation instance
automation_system = None

def get_automation_system():
    """Get or create the automation system instance."""
    global automation_system
    if automation_system is None:
        config = {
            "headless": True,
            "browser_type": "chromium",
            "max_retries": 3,
            "retry_delay": 5,
            "timeout_seconds": 300,
            "use_proxy": False
        }
        automation_system = SurveyAutomation(config)
    return automation_system

@automation_bp.route('/tasks', methods=['POST'])
@cross_origin()
def create_task():
    """Create a new survey automation task."""
    try:
        data = request.json
        
        # Validate required fields
        if not data or 'platform' not in data or 'credentials' not in data:
            return jsonify({
                'error': 'Missing required fields: platform and credentials'
            }), 400
        
        platform = data['platform']
        credentials = data['credentials']
        survey_id = data.get('survey_id')
        
        # Validate platform
        if platform.lower() not in ['swagbucks', 'inboxdollars']:
            return jsonify({
                'error': 'Unsupported platform. Use "swagbucks" or "inboxdollars"'
            }), 400
        
        # Validate credentials
        if not isinstance(credentials, dict) or 'email' not in credentials or 'password' not in credentials:
            return jsonify({
                'error': 'Credentials must include email and password'
            }), 400
        
        # Create task
        automation = get_automation_system()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            task_id = loop.run_until_complete(
                automation.create_task(platform, credentials, survey_id)
            )
            
            return jsonify({
                'task_id': task_id,
                'status': 'created',
                'message': f'Task created for {platform}'
            }), 201
            
        finally:
            loop.close()
        
    except Exception as e:
        logger.error(f"Error creating task: {e}")
        return jsonify({
            'error': 'Failed to create task',
            'details': str(e)
        }), 500

@automation_bp.route('/tasks/<task_id>/start', methods=['POST'])
@cross_origin()
def start_task(task_id):
    """Start executing a survey automation task."""
    try:
        automation = get_automation_system()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            success = loop.run_until_complete(automation.start_task(task_id))
            
            if success:
                return jsonify({
                    'task_id': task_id,
                    'status': 'started',
                    'message': 'Task execution started'
                })
            else:
                return jsonify({
                    'error': 'Failed to start task',
                    'task_id': task_id
                }), 400
                
        finally:
            loop.close()
        
    except Exception as e:
        logger.error(f"Error starting task {task_id}: {e}")
        return jsonify({
            'error': 'Failed to start task',
            'details': str(e)
        }), 500

@automation_bp.route('/tasks/<task_id>', methods=['GET'])
@cross_origin()
def get_task_status(task_id):
    """Get the status of a specific task."""
    try:
        automation = get_automation_system()
        status = automation.get_task_status(task_id)
        
        if status:
            return jsonify(status)
        else:
            return jsonify({
                'error': 'Task not found',
                'task_id': task_id
            }), 404
        
    except Exception as e:
        logger.error(f"Error getting task status {task_id}: {e}")
        return jsonify({
            'error': 'Failed to get task status',
            'details': str(e)
        }), 500

@automation_bp.route('/tasks', methods=['GET'])
@cross_origin()
def get_all_tasks():
    """Get status of all tasks."""
    try:
        automation = get_automation_system()
        tasks = automation.get_all_tasks()
        
        return jsonify({
            'tasks': tasks,
            'total': len(tasks)
        })
        
    except Exception as e:
        logger.error(f"Error getting all tasks: {e}")
        return jsonify({
            'error': 'Failed to get tasks',
            'details': str(e)
        }), 500

@automation_bp.route('/tasks/<task_id>/cancel', methods=['POST'])
@cross_origin()
def cancel_task(task_id):
    """Cancel a running task."""
    try:
        automation = get_automation_system()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            success = loop.run_until_complete(automation.cancel_task(task_id))
            
            if success:
                return jsonify({
                    'task_id': task_id,
                    'status': 'cancelled',
                    'message': 'Task cancelled successfully'
                })
            else:
                return jsonify({
                    'error': 'Failed to cancel task or task not found',
                    'task_id': task_id
                }), 400
                
        finally:
            loop.close()
        
    except Exception as e:
        logger.error(f"Error cancelling task {task_id}: {e}")
        return jsonify({
            'error': 'Failed to cancel task',
            'details': str(e)
        }), 500

@automation_bp.route('/config/proxy', methods=['POST'])
@cross_origin()
def configure_proxy():
    """Configure proxy settings."""
    try:
        data = request.json
        
        if not data or 'proxy_list' not in data:
            return jsonify({
                'error': 'Missing proxy_list in request body'
            }), 400
        
        proxy_list = data['proxy_list']
        username = data.get('username')
        password = data.get('password')
        
        if not isinstance(proxy_list, list):
            return jsonify({
                'error': 'proxy_list must be an array of proxy server addresses'
            }), 400
        
        automation = get_automation_system()
        automation.configure_proxy(proxy_list, username, password)
        
        return jsonify({
            'message': f'Configured {len(proxy_list)} proxies',
            'proxy_count': len(proxy_list)
        })
        
    except Exception as e:
        logger.error(f"Error configuring proxy: {e}")
        return jsonify({
            'error': 'Failed to configure proxy',
            'details': str(e)
        }), 500

@automation_bp.route('/config/persona', methods=['POST'])
@cross_origin()
def set_persona():
    """Set AI persona for response generation."""
    try:
        data = request.json
        
        if not data or 'persona' not in data:
            return jsonify({
                'error': 'Missing persona in request body'
            }), 400
        
        persona_name = data['persona']
        
        automation = get_automation_system()
        success = automation.set_ai_persona(persona_name)
        
        if success:
            return jsonify({
                'message': f'Set AI persona to {persona_name}',
                'persona': persona_name
            })
        else:
            available_personas = automation.get_available_personas()
            return jsonify({
                'error': f'Persona "{persona_name}" not found',
                'available_personas': available_personas
            }), 400
        
    except Exception as e:
        logger.error(f"Error setting persona: {e}")
        return jsonify({
            'error': 'Failed to set persona',
            'details': str(e)
        }), 500

@automation_bp.route('/config/personas', methods=['GET'])
@cross_origin()
def get_personas():
    """Get available AI personas."""
    try:
        automation = get_automation_system()
        personas = automation.get_available_personas()
        
        return jsonify({
            'personas': personas,
            'total': len(personas)
        })
        
    except Exception as e:
        logger.error(f"Error getting personas: {e}")
        return jsonify({
            'error': 'Failed to get personas',
            'details': str(e)
        }), 500

@automation_bp.route('/health', methods=['GET'])
@cross_origin()
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'survey-automation',
        'version': '1.0.0'
    })

@automation_bp.route('/platforms', methods=['GET'])
@cross_origin()
def get_supported_platforms():
    """Get list of supported platforms."""
    return jsonify({
        'platforms': [
            {
                'name': 'swagbucks',
                'display_name': 'Swagbucks',
                'url': 'https://www.swagbucks.com',
                'supported_features': [
                    'login_automation',
                    'survey_navigation',
                    'question_extraction',
                    'ai_responses',
                    'captcha_solving'
                ]
            },
            {
                'name': 'inboxdollars',
                'display_name': 'InboxDollars',
                'url': 'https://www.inboxdollars.com',
                'supported_features': [
                    'login_automation',
                    'survey_navigation',
                    'question_extraction',
                    'ai_responses',
                    'captcha_solving'
                ]
            }
        ]
    })

