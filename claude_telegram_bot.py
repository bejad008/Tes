# pip install python-telegram-bot llama-cpp-python requests beautifulsoup4

import logging
import os
import sys
import asyncio
import threading
import json
import tempfile
import subprocess
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from telegram import Update, Document
from telegram.ext import Application, MessageHandler, filters, ContextTypes

# Logging setup
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Configuration
BOT_TOKEN = "8022320384:AAFkG2EvVr0RMqdLeQAK54Jl-oEClbQeMOU"  # Replace with your actual token
WORKSPACE_DIR = "bot_workspace"

# Global variables
llm_model = None
model_loaded = False
model_lock = threading.Lock()
executor = ThreadPoolExecutor(max_workers=3)

# Create workspace directory
Path(WORKSPACE_DIR).mkdir(exist_ok=True)

def load_ai_model():
    """Load the Qwen programming model"""
    global llm_model, model_loaded
    
    try:
        from llama_cpp import Llama
        
        print("🚀 Loading Claude-style AI Model...")
        
        try:
            llm_model = Llama.from_pretrained(
                repo_id="mradermacher/Qwen3-4B-Claude-Sonnet-4-Reasoning-Distill-Safetensor-GGUF",
                filename="Qwen3-4B-Claude-Sonnet-4-Reasoning-Distill-Safetensor.Q6_K.gguf",
                local_files_only=False,
                verbose=False,
                n_ctx=8192,         # Large context for conversations
                n_threads=6,        
                n_batch=64,
                use_mlock=True,     
                use_mmap=True,      
                n_gpu_layers=0,     
                seed=42,            
                f16_kv=True,        
            )
            
            print("✅ Claude-style model loaded successfully!")
            model_loaded = True
            return True
            
        except Exception as e:
            print(f"⚠️ Failed to load Q6_K model: {e}")
            try:
                print("🔄 Trying Q4_K_M model...")
                llm_model = Llama.from_pretrained(
                    repo_id="mradermacher/Qwen3-4B-Claude-Sonnet-4-Reasoning-Distill-Safetensor-GGUF",
                    filename="Qwen3-4B-Claude-Sonnet-4-Reasoning-Distill-Safetensor.Q4_K_M.gguf",
                    n_ctx=4096,
                    n_threads=4,
                    n_batch=32,
                    use_mlock=True,
                    use_mmap=True,
                    verbose=False
                )
                model_loaded = True
                print("✅ Q4_K_M fallback model loaded!")
                return True
            except Exception as e2:
                print(f"⚠️ All models failed: {e2}")
                print("⚡ Running in fallback mode")
                model_loaded = False
                return False
            
    except ImportError:
        print("⚠️ llama-cpp-python not installed. Running in fallback mode.")
        return False

def get_user_workspace(user_id: int) -> Path:
    """Get user's workspace directory"""
    user_dir = Path(WORKSPACE_DIR) / str(user_id)
    user_dir.mkdir(exist_ok=True)
    return user_dir

def detect_user_intent(message: str) -> dict:
    """Detect what user wants to do based on their message"""
    message_lower = message.lower().strip()
    
    # Code generation requests
    code_keywords = [
        'create', 'make', 'build', 'generate', 'write', 'code', 'program', 
        'script', 'function', 'class', 'algorithm', 'implement', 'develop'
    ]
    
    # File operations
    save_keywords = ['save', 'store', 'keep', 'remember this', 'save this code']
    read_keywords = ['show me', 'read', 'open', 'display', 'view', 'see the file']
    list_keywords = ['list', 'show files', 'my files', 'what files']
    run_keywords = ['run', 'execute', 'test', 'try', 'play']
    edit_keywords = ['edit', 'modify', 'change', 'update', 'fix']
    delete_keywords = ['delete', 'remove', 'erase']
    
    # Error analysis
    error_keywords = ['error', 'bug', 'problem', 'issue', 'fix', 'debug', 'not working', 'broken']
    review_keywords = ['review', 'check', 'analyze', 'improve', 'optimize']
    
    # Analyze message
    intent = {
        'type': 'general',
        'action': None,
        'filename': None,
        'content': message
    }
    
    # Extract filename if present
    filename_match = re.search(r'(\w+\.py)', message)
    if filename_match:
        intent['filename'] = filename_match.group(1)
    
    # Determine intent
    if any(keyword in message_lower for keyword in error_keywords):
        intent['type'] = 'error_analysis'
    elif any(keyword in message_lower for keyword in review_keywords):
        intent['type'] = 'code_review'
    elif any(keyword in message_lower for keyword in save_keywords):
        intent['type'] = 'file_save'
    elif any(keyword in message_lower for keyword in read_keywords):
        intent['type'] = 'file_read'
    elif any(keyword in message_lower for keyword in list_keywords):
        intent['type'] = 'file_list'
    elif any(keyword in message_lower for keyword in run_keywords):
        intent['type'] = 'file_run'
    elif any(keyword in message_lower for keyword in edit_keywords):
        intent['type'] = 'file_edit'
    elif any(keyword in message_lower for keyword in delete_keywords):
        intent['type'] = 'file_delete'
    elif any(keyword in message_lower for keyword in code_keywords):
        intent['type'] = 'code_generation'
    
    return intent

def analyze_python_error(message: str, code: str) -> str:
    """Analyze Python errors and provide solutions"""
    message_lower = message.lower()
    
    common_errors = {
        'indentationerror': {
            'issue': 'Incorrect indentation in your Python code',
            'solution': 'Make sure all code blocks are properly indented with 4 spaces or consistent tabs',
            'example': 'if True:\n    print("Correct indentation")'
        },
        'syntaxerror': {
            'issue': 'Syntax error in your Python code',
            'solution': 'Check for missing colons, parentheses, or quotes',
            'example': 'if condition:  # Don\'t forget the colon'
        },
        'nameerror': {
            'issue': 'Variable or function name not defined',
            'solution': 'Make sure all variables are defined before use',
            'example': 'variable = "value"  # Define before using'
        },
        'typeerror': {
            'issue': 'Type mismatch or incorrect operation',
            'solution': 'Check that you\'re using compatible data types',
            'example': 'str(number) + " text"  # Convert types when needed'
        },
        'indexerror': {
            'issue': 'List index out of range',
            'solution': 'Check list length before accessing elements',
            'example': 'if len(my_list) > index:\n    value = my_list[index]'
        },
        'keyerror': {
            'issue': 'Dictionary key not found',
            'solution': 'Use .get() method or check if key exists',
            'example': 'value = my_dict.get("key", "default")'
        },
        'importerror': {
            'issue': 'Module import failed',
            'solution': 'Install required packages or check module names',
            'example': 'pip install package_name'
        }
    }
    
    analysis = "**🔍 Error Analysis:**\n\n"
    
    for error_type, info in common_errors.items():
        if error_type in message_lower:
            analysis += f"**Issue:** {info['issue']}\n\n"
            analysis += f"**Solution:** {info['solution']}\n\n"
            analysis += f"**Example:**\n```python\n{info['example']}\n```\n\n"
            break
    else:
        analysis += "I can help you debug this issue! Please share:\n"
        analysis += "• The complete error message\n"
        analysis += "• The code that's causing the problem\n"
        analysis += "• What you expected to happen\n\n"
        analysis += "This will help me provide a more specific solution."
    
    return analysis

def analyze_code_for_improvements(code: str) -> str:
    """Analyze code and suggest improvements"""
    suggestions = []
    
    # Check for common improvements
    if 'print(' in code and 'f"' not in code and '".format(' not in code:
        suggestions.append("Consider using f-strings for better string formatting: f'Hello {name}'")
    
    if 'try:' not in code and ('open(' in code or 'requests.' in code):
        suggestions.append("Add error handling with try-except blocks for file operations or network requests")
    
    if re.search(r'def \w+\(.*\):', code) and '"""' not in code:
        suggestions.append("Add docstrings to your functions to document what they do")
    
    if 'input(' in code and 'try:' not in code:
        suggestions.append("Add input validation and error handling for user input")
    
    if len([line for line in code.split('\n') if line.strip()]) > 20:
        suggestions.append("Consider breaking long functions into smaller, more focused functions")
    
    if suggestions:
        return "• " + "\n• ".join(suggestions)
    else:
        return "Your code looks good! It follows Python best practices."

def fix_common_code_issues(code: str) -> str:
    """Fix common code issues automatically"""
    lines = code.split('\n')
    fixed_lines = []
    
    for line in lines:
        # Fix basic indentation issues (convert tabs to spaces)
        line = line.replace('\t', '    ')
        
        # Add missing colons for control structures
        if re.match(r'^\s*(if|elif|else|for|while|def|class)\s+.*[^:]$', line.strip()):
            line = line.rstrip() + ':'
        
        fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

def generate_claude_response(prompt: str, context: str = "") -> str:
    """Generate Claude-style response"""
    global llm_model, model_loaded
    
    if not model_loaded or not llm_model:
        return generate_fallback_response(prompt)
    
    try:
        # Claude-style system prompt
        system_prompt = """You are Claude, an AI assistant created by Anthropic. You are helpful, harmless, and honest. You excel at:

- Programming and coding in all languages
- Explaining complex concepts clearly
- Writing and creative tasks
- Analysis and problem-solving
- Providing detailed, thoughtful responses

You should respond naturally and conversationally, like the real Claude. When asked to create code, provide clean, well-commented, working code. When asked questions, give comprehensive but concise answers.

Always be helpful and aim to fully address what the user is asking for."""

        full_prompt = f"{system_prompt}\n\nHuman: {prompt}\n\nAssistant:"
        
        with model_lock:
            response = llm_model(
                full_prompt,
                max_tokens=2000,
                temperature=0.7,
                top_p=0.9,
                top_k=40,
                stop=["Human:", "User:", "\n\nHuman:", "\n\nUser:"],
                echo=False,
                repeat_penalty=1.1,
            )
        
        if response and 'choices' in response and len(response['choices']) > 0:
            response_text = response['choices'][0]['text'].strip()
            
            if response_text:
                # Clean up response
                response_text = response_text.replace("Assistant:", "").strip()
                
                if len(response_text) > 4000:  # Telegram message limit
                    response_text = response_text[:4000] + "\n\n... (response truncated)"
                
                return response_text
        
        return generate_fallback_response(prompt)
        
    except Exception as e:
        logger.error(f"Response generation error: {e}")
        return generate_fallback_response(prompt)

def generate_fallback_response(prompt: str) -> str:
    """Generate fallback response when AI model fails"""
    prompt_lower = prompt.lower()
    
    if any(word in prompt_lower for word in ['calculator', 'math', 'compute']):
        return """I can help you create a calculator! Here's a simple Python calculator:

```python
def calculator():
    print("Simple Calculator")
    
    try:
        num1 = float(input("Enter first number: "))
        operator = input("Enter operator (+, -, *, /): ")
        num2 = float(input("Enter second number: "))
        
        if operator == '+':
            result = num1 + num2
        elif operator == '-':
            result = num1 - num2
        elif operator == '*':
            result = num1 * num2
        elif operator == '/':
            if num2 != 0:
                result = num1 / num2
            else:
                return "Error: Division by zero!"
        else:
            return "Error: Invalid operator!"
        
        return f"Result: {result}"
    
    except ValueError:
        return "Error: Invalid input!"

# Run the calculator
if __name__ == "__main__":
    print(calculator())
```

This calculator handles basic arithmetic operations with error handling for invalid inputs and division by zero."""

    elif any(word in prompt_lower for word in ['web', 'scraper', 'scraping', 'requests']):
        return """I'll help you create a web scraper! Here's a Python web scraper example:

```python
import requests
from bs4 import BeautifulSoup
import json
import time

def web_scraper(url):
    \"\"\"Simple web scraper with error handling\"\"\"
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract basic information
        title = soup.find('title')
        title_text = title.get_text().strip() if title else "No title found"
        
        # Extract all paragraphs
        paragraphs = [p.get_text().strip() for p in soup.find_all('p') if p.get_text().strip()]
        
        # Extract all links
        links = [a.get('href') for a in soup.find_all('a', href=True)]
        
        return {
            'url': url,
            'title': title_text,
            'paragraphs': paragraphs[:5],  # First 5 paragraphs
            'links_count': len(links),
            'links': links[:10]  # First 10 links
        }
    
    except requests.RequestException as e:
        return f"Error fetching URL: {str(e)}"
    except Exception as e:
        return f"Error parsing content: {str(e)}"

# Example usage
if __name__ == "__main__":
    result = web_scraper("https://example.com")
    print(json.dumps(result, indent=2))
```

This scraper includes proper headers, error handling, and extracts common web page elements."""

    elif any(word in prompt_lower for word in ['hello', 'hi', 'hey', 'greet']):
        return """Hello! I'm Claude, an AI assistant created by Anthropic. I'm here to help you with a wide variety of tasks including:

• Programming and coding in any language
• Writing and creative projects
• Analysis and problem-solving
• Answering questions and explaining concepts
• File management and code execution

You can talk to me naturally - just tell me what you need help with! For example:
- "Create a Python script that..."
- "Explain how machine learning works"
- "Help me debug this code"
- "Write a story about..."

What would you like to work on today?"""

    else:
        return f"""I understand you're asking about: {prompt[:100]}...

I'm Claude, an AI assistant, and I'd be happy to help you with this! However, I'm currently running in a simplified mode. 

For programming tasks, I can help you:
- Write Python code for various purposes
- Create scripts and applications  
- Debug and improve existing code
- Explain programming concepts

For other topics, I can:
- Answer questions and provide explanations
- Help with writing and analysis
- Discuss ideas and provide insights

Could you provide a bit more detail about what specifically you'd like me to help you with? I'll do my best to give you a comprehensive and helpful response."""

async def save_file(user_id: int, filename: str, content: str) -> bool:
    """Save file to user's workspace"""
    try:
        user_dir = get_user_workspace(user_id)
        file_path = user_dir / filename
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return True
    except Exception as e:
        logger.error(f"File save error: {e}")
        return False

async def read_file(user_id: int, filename: str) -> str:
    """Read file from user's workspace"""
    try:
        user_dir = get_user_workspace(user_id)
        file_path = user_dir / filename
        
        if not file_path.exists():
            return None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"File read error: {e}")
        return None

async def list_files(user_id: int) -> list:
    """List files in user's workspace"""
    try:
        user_dir = get_user_workspace(user_id)
        return [f.name for f in user_dir.iterdir() if f.is_file()]
    except Exception as e:
        logger.error(f"File list error: {e}")
        return []

async def execute_python_file(user_id: int, filename: str) -> str:
    """Execute Python file safely"""
    try:
        user_dir = get_user_workspace(user_id)
        file_path = user_dir / filename
        
        if not file_path.exists():
            return f"I couldn't find the file '{filename}' in your workspace."
        
        result = subprocess.run(
            [sys.executable, str(file_path)],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=user_dir
        )
        
        output = ""
        if result.stdout:
            output += f"Output:\n{result.stdout}\n"
        if result.stderr:
            output += f"Errors:\n{result.stderr}\n"
        
        if not output:
            output = "The script executed successfully with no output."
        
        return output[:2000]
        
    except subprocess.TimeoutExpired:
        return "The execution timed out after 10 seconds."
    except Exception as e:
        return f"There was an error executing the file: {str(e)}"

def extract_code_from_message(message: str) -> str:
    """Extract code blocks from message"""
    # Look for code blocks
    code_blocks = re.findall(r'```(?:python)?\n?(.*?)\n?```', message, re.DOTALL)
    if code_blocks:
        return code_blocks[0].strip()
    
    # Look for inline code
    inline_code = re.findall(r'`([^`]+)`', message)
    if inline_code:
        return '\n'.join(inline_code)
    
    return None

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle all messages like Claude"""
    try:
        user_message = update.message.text
        user_id = update.effective_user.id
        
        # Skip very short messages
        if len(user_message.strip()) < 2:
            return
        
        await context.bot.send_chat_action(update.effective_chat.id, "typing")
        
        # Detect user intent
        intent = detect_user_intent(user_message)
        
        # Handle error analysis and debugging
        if intent['type'] == 'error_analysis':
            code = extract_code_from_message(user_message)
            analysis = analyze_python_error(user_message, code or "")
            
            if code:
                # Try to fix common issues
                fixed_code = fix_common_code_issues(code)
                if fixed_code != code:
                    analysis += f"\n\n**🔧 Fixed Code:**\n```python\n{fixed_code}\n```"
                
                # Additional suggestions
                improvements = analyze_code_for_improvements(code)
                analysis += f"\n\n**💡 Additional Suggestions:**\n{improvements}"
            
            await update.message.reply_text(analysis, parse_mode='Markdown')
            return
        
        elif intent['type'] == 'code_review':
            code = extract_code_from_message(user_message)
            if code:
                improvements = analyze_code_for_improvements(code)
                response = f"**📋 Code Review:**\n\n{improvements}\n\n**🔧 Optimized Version:**\n```python\n{fix_common_code_issues(code)}\n```"
            else:
                response = "I'd be happy to review your code! Please share the code you'd like me to analyze, and I'll provide suggestions for improvements, best practices, and potential optimizations."
            
            await update.message.reply_text(response, parse_mode='Markdown')
            return
        
        # Handle file operations naturally
        if intent['type'] == 'file_save':
            code = extract_code_from_message(user_message)
            if code and intent['filename']:
                success = await save_file(user_id, intent['filename'], code)
                if success:
                    response = f"I've saved your code to {intent['filename']}. The file is now in your workspace and ready to use."
                else:
                    response = f"I encountered an issue saving the file {intent['filename']}. Please try again."
            else:
                response = "I'd be happy to save your code! Please provide the code and specify a filename (e.g., 'script.py')."
            
            await update.message.reply_text(response)
            return
        
        elif intent['type'] == 'file_read':
            if intent['filename']:
                content = await read_file(user_id, intent['filename'])
                if content:
                    if len(content) > 3000:
                        content = content[:3000] + "\n\n... (file truncated for display)"
                    response = f"Here's the content of {intent['filename']}:\n\n```python\n{content}\n```"
                else:
                    response = f"I couldn't find {intent['filename']} in your workspace. Would you like me to list your available files?"
            else:
                files = await list_files(user_id)
                if files:
                    file_list = '\n'.join([f"• {f}" for f in sorted(files)])
                    response = f"Here are the files in your workspace:\n\n{file_list}\n\nWhich file would you like me to show you?"
                else:
                    response = "Your workspace is currently empty. You can create files by asking me to write code and save it."
            
            await update.message.reply_text(response)
            return
        
        elif intent['type'] == 'file_list':
            files = await list_files(user_id)
            if files:
                file_list = '\n'.join([f"• {f}" for f in sorted(files)])
                response = f"Here are your files ({len(files)} total):\n\n{file_list}\n\nI can help you read, edit, or run any of these files."
            else:
                response = "Your workspace is currently empty. When you create or save code, the files will appear here."
            
            await update.message.reply_text(response)
            return
        
        elif intent['type'] == 'file_run':
            if intent['filename']:
                result = await execute_python_file(user_id, intent['filename'])
                response = f"Here's the result of running {intent['filename']}:\n\n```\n{result}\n```"
            else:
                files = await list_files(user_id)
                py_files = [f for f in files if f.endswith('.py')]
                if py_files:
                    file_list = '\n'.join([f"• {f}" for f in py_files])
                    response = f"Which Python file would you like me to run?\n\n{file_list}"
                else:
                    response = "You don't have any Python files to run yet. Create some code first and I'll help you execute it."
            
            await update.message.reply_text(response)
            return
        
        # Generate Claude-style response
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(executor, generate_claude_response, user_message)
        
        # Send response
        await update.message.reply_text(response)
        
    except Exception as e:
        logger.error(f"Message handling error: {e}")
        await update.message.reply_text("I encountered an error processing your message. Please try rephrasing your request.")

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle uploaded files naturally"""
    try:
        document = update.message.document
        user_id = update.effective_user.id
        
        if document.file_name.endswith('.py'):
            # Download and save Python file
            file = await context.bot.get_file(document.file_id)
            file_content = await file.download_as_bytearray()
            
            success = await save_file(user_id, document.file_name, file_content.decode('utf-8'))
            
            if success:
                response = f"I've received and saved your Python file '{document.file_name}' to your workspace. I can help you read it, run it, or modify it - just let me know what you'd like to do!"
            else:
                response = f"I had trouble saving your file '{document.file_name}'. Please try uploading it again."
        else:
            response = f"I received your file '{document.file_name}'. Currently, I work best with Python (.py) files. If you have Python code to share, please save it as a .py file and upload it again."
        
        await update.message.reply_text(response)
        
    except Exception as e:
        logger.error(f"Document handling error: {e}")
        await update.message.reply_text("I had trouble processing your file. Please try uploading it again.")

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    """Handle errors gracefully"""
    logger.error(f"Error: {context.error}")

def main():
    """Main function"""
    print("=" * 60)
    print("🤖 CLAUDE-STYLE TELEGRAM BOT 🤖")
    print("=" * 60)
    
    # Load model
    print("🧠 Loading AI model...")
    model_success = load_ai_model()
    
    # Setup bot
    print("⚡ Setting up Claude-style bot...")
    app = Application.builder().token(BOT_TOKEN).build()
    
    # Add handlers - NO COMMANDS, only natural conversation
    app.add_handler(MessageHandler(filters.TEXT, handle_message))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_error_handler(error_handler)
    
    print("\n" + "=" * 60)
    print("🎯 CLAUDE-STYLE BOT READY! 🎯")
    print("=" * 60)
    print(f"🧠 AI Model: {'🚀 Advanced Loaded' if model_success else '⚡ Fallback Mode'}")
    print(f"💬 Natural Conversation: ✅ READY")
    print(f"🐍 Python Support: ✅ READY") 
    print(f"📁 File Management: ✅ READY")
    print(f"▶️ Code Execution: ✅ READY")
    print("🚫 NO COMMANDS - Just chat naturally!")
    print("🛑 Press Ctrl+C to stop")
    print("=" * 60)
    
    try:
        app.run_polling(drop_pending_updates=True)
    except KeyboardInterrupt:
        print("\n🎯 Claude-style bot stopped!")
    except Exception as e:
        print(f"⚠️ Bot error: {e}")
    finally:
        executor.shutdown(wait=False)

if __name__ == "__main__":
    main()
