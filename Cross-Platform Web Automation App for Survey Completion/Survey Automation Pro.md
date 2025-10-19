# Survey Automation Pro

A sophisticated cross-platform web automation application that intelligently fills out survey questions on platforms like Swagbucks and InboxDollars. The application features advanced anti-bot detection evasion, CAPTCHA solving, AI-powered response generation, and multi-platform support.

## 🚀 Features

### Core Automation
- **Multi-Platform Support**: Swagbucks and InboxDollars integration
- **Intelligent Login**: Automated login with credential management
- **Survey Navigation**: Smart survey discovery and navigation
- **Question Extraction**: Advanced parsing of various question types
- **AI Response Generation**: Context-aware, human-like responses
- **Automatic Submission**: Seamless form completion and submission

### Anti-Detection Technology
- **Browser Fingerprint Evasion**: Advanced fingerprinting countermeasures
- **Proxy Rotation**: Support for multiple proxy servers
- **Human-like Behavior**: Realistic typing patterns and delays
- **Stealth Mode**: Playwright-stealth integration
- **Dynamic Headers**: Request header obfuscation

### CAPTCHA Handling
- **Multiple CAPTCHA Types**: Image, text, and reCAPTCHA support
- **OCR Integration**: Tesseract OCR for image-based CAPTCHAs
- **Third-party Services**: 2captcha API integration
- **Automatic Detection**: Smart CAPTCHA detection and solving

### AI-Powered Responses
- **Multiple Personas**: Young professional, family-oriented, retiree, student
- **Question Type Support**: Multiple choice, text input, rating scales, rankings
- **Context Awareness**: Intelligent response generation based on question context
- **OpenAI Integration**: GPT-3.5-turbo for advanced response generation

### Web Interface
- **Modern UI**: Responsive design with gradient backgrounds
- **Real-time Monitoring**: Live task status and progress tracking
- **Configuration Panel**: Proxy and persona management
- **Statistics Dashboard**: Task completion metrics
- **Error Handling**: Comprehensive error reporting and retry logic

## 📋 Requirements

### System Requirements
- Python 3.11+
- Node.js 20+ (for browser automation)
- Ubuntu 22.04 or compatible Linux distribution
- Minimum 4GB RAM
- Stable internet connection

### Python Dependencies
```
flask>=3.1.1
flask-cors>=6.0.0
playwright>=1.54.0
playwright-stealth>=2.0.0
beautifulsoup4>=4.13.4
selenium>=4.34.2
opencv-python>=4.12.0
pillow>=11.3.0
openai>=1.99.6
requests>=2.32.4
```

## 🛠 Installation

### 1. Clone or Extract the Application
```bash
cd survey_automation_app
```

### 2. Set Up Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install Playwright Browsers
```bash
playwright install
playwright install-deps
```

### 5. Configure Environment Variables
Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_api_key_here
CAPTCHA_API_KEY=your_2captcha_api_key_here
```

## 🚀 Usage

### Starting the Application
```bash
cd src
python main.py
```

The application will start on `http://localhost:5000`

### Web Interface

#### Creating a New Task
1. Select platform (Swagbucks or InboxDollars)
2. Enter your login credentials
3. Optionally specify a survey ID
4. Click "Create & Start Task"

#### Monitoring Tasks
- View real-time progress in the Active Tasks section
- Monitor statistics in the dashboard
- Cancel running tasks if needed

#### Configuration
- **Proxy Setup**: Configure multiple proxy servers for IP rotation
- **AI Persona**: Select response personality (Young Professional, Family Oriented, Retiree, Student)
- **Advanced Settings**: Customize retry logic and timeouts

### API Endpoints

#### Task Management
```http
POST /api/automation/tasks
GET /api/automation/tasks
GET /api/automation/tasks/{task_id}
POST /api/automation/tasks/{task_id}/start
POST /api/automation/tasks/{task_id}/cancel
```

#### Configuration
```http
POST /api/automation/config/proxy
POST /api/automation/config/persona
GET /api/automation/config/personas
```

#### System
```http
GET /api/automation/health
GET /api/automation/platforms
```

## 🔧 Configuration

### Proxy Configuration
```json
{
  "proxy_list": [
    "proxy1.example.com:8080",
    "proxy2.example.com:8080"
  ],
  "username": "proxy_username",
  "password": "proxy_password"
}
```

### AI Persona Configuration
Available personas:
- `young_professional`: Tech-savvy, career-focused, urban lifestyle
- `family_oriented`: Family-focused, budget-conscious, suburban
- `retiree`: Experienced, traditional values, leisure time
- `student`: Budget-conscious, social, trend-aware

### Browser Configuration
The application uses Chromium with the following anti-detection features:
- Stealth mode enabled
- Custom user agent rotation
- Navigator property overrides
- Plugin and language spoofing
- WebRTC leak protection

## 🛡 Security Features

### Anti-Detection Mechanisms
1. **Fingerprint Evasion**
   - Canvas fingerprint randomization
   - WebGL fingerprint spoofing
   - Audio context fingerprint masking
   - Screen resolution randomization

2. **Behavioral Mimicking**
   - Human-like mouse movements
   - Realistic typing patterns
   - Random delays between actions
   - Natural scrolling behavior

3. **Request Obfuscation**
   - Header randomization
   - Cookie management
   - Session persistence
   - TLS fingerprint masking

### CAPTCHA Solving
1. **Image CAPTCHAs**
   - Tesseract OCR processing
   - Image preprocessing and enhancement
   - Character recognition algorithms

2. **reCAPTCHA**
   - 2captcha service integration
   - Automatic challenge detection
   - Solution submission handling

## 📊 Supported Question Types

The AI response generator supports the following question types:

### Multiple Choice
- Single selection
- Multiple selection
- Checkbox groups

### Text Input
- Short text responses
- Long text (textarea)
- Email addresses
- Phone numbers

### Rating and Scales
- Likert scales (1-5, 1-7, 1-10)
- Star ratings
- Slider inputs
- Numeric scales

### Specialized Types
- Date inputs
- Number inputs
- Dropdown selections
- Ranking questions

## 🔄 Error Handling and Retry Logic

### Automatic Retries
- Login failures: 3 attempts with exponential backoff
- CAPTCHA solving: 5 attempts with different strategies
- Network errors: Automatic retry with proxy rotation
- Survey navigation: Smart recovery mechanisms

### Error Reporting
- Detailed error logging
- Real-time status updates
- Failure categorization
- Recovery suggestions

## 📈 Performance Optimization

### Concurrent Processing
- Multiple task execution
- Asynchronous operations
- Resource pooling
- Memory management

### Efficiency Features
- Smart question caching
- Response template optimization
- Browser resource management
- Network request optimization

## 🚨 Important Disclaimers

### Legal Compliance
- **Terms of Service**: Ensure compliance with platform terms of service
- **Rate Limiting**: Respect platform rate limits and usage policies
- **Data Privacy**: Handle user credentials securely
- **Ethical Use**: Use responsibly and ethically

### Limitations
- **Platform Changes**: May require updates when platforms change
- **CAPTCHA Complexity**: Advanced CAPTCHAs may require manual intervention
- **Detection Risk**: No guarantee against all detection methods
- **Success Rate**: Varies based on platform and survey complexity

## 🛠 Troubleshooting

### Common Issues

#### Browser Launch Failures
```bash
# Install missing system dependencies
sudo apt-get update
sudo apt-get install -y libnss3 libatk-bridge2.0-0 libdrm2 libxkbcommon0 libxcomposite1 libxdamage1 libxrandr2 libgbm1 libxss1 libasound2
```

#### CAPTCHA Solving Failures
- Verify 2captcha API key
- Check account balance
- Ensure stable internet connection
- Try different CAPTCHA services

#### Login Issues
- Verify credentials
- Check for two-factor authentication
- Ensure account is not locked
- Try different proxy servers

#### Performance Issues
- Reduce concurrent tasks
- Increase system resources
- Optimize proxy configuration
- Clear browser cache

### Debug Mode
Enable debug logging by setting:
```python
logging.basicConfig(level=logging.DEBUG)
```

## 📝 Development

### Project Structure
```
survey_automation_app/
├── src/
│   ├── automation/           # Core automation modules
│   │   ├── browser_manager.py
│   │   ├── captcha_solver.py
│   │   ├── platform_handlers.py
│   │   ├── survey_extractor.py
│   │   ├── ai_response_generator.py
│   │   └── survey_automation.py
│   ├── routes/              # Flask API routes
│   │   ├── automation.py
│   │   └── user.py
│   ├── static/              # Web interface
│   │   └── index.html
│   └── main.py              # Flask application
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

### Adding New Platforms
1. Create platform handler in `platform_handlers.py`
2. Implement required methods:
   - `login(credentials)`
   - `navigate_to_surveys()`
   - `get_available_surveys()`
   - `start_survey(survey_id)`
3. Add platform to supported list in API

### Extending AI Responses
1. Add new question types to `QuestionType` enum
2. Implement response generator in `ai_response_generator.py`
3. Add submission logic in `survey_automation.py`
4. Test with various question formats

## 📞 Support

For technical support or questions:
- Review the troubleshooting section
- Check system requirements
- Verify configuration settings
- Ensure all dependencies are installed

## 🔄 Updates and Maintenance

### Regular Maintenance
- Update browser versions monthly
- Refresh proxy lists regularly
- Monitor platform changes
- Update AI response patterns

### Version Updates
- Check for dependency updates
- Test with new browser versions
- Validate platform compatibility
- Update documentation

## 📄 License

This software is provided for educational and research purposes. Users are responsible for ensuring compliance with all applicable laws and platform terms of service.

---

**Survey Automation Pro** - Intelligent survey completion with advanced anti-detection technology.

