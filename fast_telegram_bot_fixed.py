# pip install python-telegram-bot llama-cpp-python requests pillow huggingface_hub

import logging
import os
import sys
import asyncio
import threading
import requests
import io
import base64
from concurrent.futures import ThreadPoolExecutor
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Fast logging setup
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.WARNING  # Reduced log level for performance
)
logger = logging.getLogger(__name__)

# Configuration
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8305182212:AAEDWgzi09qYZMMZDj-o-H-NX5jdvu3cm9E")
HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN", "")  # Add your HuggingFace token here

# Global variables
llm_model = None
model_loaded = False
model_lock = threading.Lock()
executor = ThreadPoolExecutor(max_workers=5)  # Increased for image generation

def load_ai_model():
    """Load AI model with SPEED optimizations"""
    global llm_model, model_loaded
    
    try:
        from llama_cpp import Llama
        
        print("🚀 Loading FAST AI model...")
        
        try:
            # Try Q6_K model first (better quality for coding)
            llm_model = Llama.from_pretrained(
                repo_id="mradermacher/SEX_ROLEPLAY-3.2-1B-i1-GGUF",
                filename="SEX_ROLEPLAY-3.2-1B.i1-Q6_K.gguf",
                local_files_only=False,
                verbose=False,
                
                # OPTIMIZED FOR CODING
                n_ctx=2048,         # Larger context for code
                n_threads=4,        
                n_batch=32,         # Larger batch for code generation
                use_mlock=True,     
                use_mmap=True,      
                n_gpu_layers=0,     
                
                seed=42,            
                f16_kv=True,        
            )
            
            # Quick test
            print("⚡ Testing model...")
            test_response = llm_model("Hello", max_tokens=5, temperature=0.1)
            print("✅ FAST AI model Q6_K loaded successfully!")
            model_loaded = True
            return True
            
        except Exception as e:
            print(f"❌ Failed to load Q6_K model: {e}")
            try:
                print("🔄 Trying Q2_K model...")
                llm_model = Llama.from_pretrained(
                    repo_id="mradermacher/SEX_ROLEPLAY-3.2-1B-i1-GGUF",
                    filename="SEX_ROLEPLAY-3.2-1B.i1-Q2_K.gguf",
                    n_ctx=1024,      
                    n_threads=4,
                    n_batch=16,
                    use_mlock=True,
                    use_mmap=True,
                    verbose=False
                )
                model_loaded = True
                print("✅ Q2_K fallback model loaded!")
                return True
            except Exception as e2:
                print(f"⚠️ All models failed: {e2}")
                print("⚡ Running in FAST echo mode")
                model_loaded = False
                return False
            
    except ImportError:
        print("❌ llama-cpp-python not installed. Running in echo mode.")
        return False

def detect_request_type(message: str) -> str:
    """Detect if user wants image, code, or chat"""
    message_lower = message.lower()
    
    # Image generation keywords
    image_keywords = ['gambar', 'image', 'foto', 'picture', 'draw', 'create image', 'generate image', 'buatkan gambar', 'bikinin gambar']
    if any(keyword in message_lower for keyword in image_keywords):
        return "image"
    
    # Coding keywords
    code_keywords = ['code', 'coding', 'program', 'script', 'function', 'python', 'javascript', 'html', 'css', 'java', 'c++', 'buatkan kode', 'bikinin code']
    if any(keyword in message_lower for keyword in code_keywords):
        return "code"
    
    return "chat"

async def generate_image(prompt: str) -> bytes:
    """Generate image using HuggingFace API"""
    try:
        # Using Stable Diffusion via HuggingFace
        API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
        headers = {"Authorization": f"Bearer {HUGGINGFACE_TOKEN}"} if HUGGINGFACE_TOKEN else {}
        
        def query():
            payload = {
                "inputs": prompt,
                "parameters": {
                    "num_inference_steps": 20,
                    "guidance_scale": 7.5,
                    "width": 512,
                    "height": 512
                }
            }
            response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
            return response.content
        
        loop = asyncio.get_event_loop()
        image_bytes = await loop.run_in_executor(executor, query)
        
        return image_bytes
        
    except Exception as e:
        logger.error(f"Image generation error: {e}")
        return None

def generate_code_response(prompt: str) -> str:
    """Generate code using AI model with special coding prompt"""
    global llm_model, model_loaded
    
    if not model_loaded or not llm_model:
        return generate_fallback_code(prompt)
    
    try:
        # Special coding prompt template
        coding_prompt = f"""You are an expert programmer. Write clean, working code for this request:

Request: {prompt}

Code:
```"""
        
        with model_lock:
            response = llm_model(
                coding_prompt,
                max_tokens=500,     # More tokens for code
                temperature=0.1,    # Low temperature for precise code
                top_p=0.9,
                top_k=40,
                stop=["```", "Request:", "Human:", "\n\n\n"],
                echo=False,
                repeat_penalty=1.1,
            )
        
        if response and 'choices' in response and len(response['choices']) > 0:
            code_text = response['choices'][0]['text'].strip()
            
            if code_text:
                # Format code response
                if len(code_text) > 2000:  # Telegram message limit consideration
                    code_text = code_text[:2000] + "\n\n... (truncated)"
                
                return f"```\n{code_text}\n```"
        
        return generate_fallback_code(prompt)
        
    except Exception as e:
        logger.error(f"Code generation error: {e}")
        return generate_fallback_code(prompt)

def generate_fallback_code(prompt: str) -> str:
    """Generate simple code examples when AI model fails"""
    prompt_lower = prompt.lower()
    
    if 'python' in prompt_lower:
        return """```python
# Python example
def hello_world():
    print("Hello, World!")
    return "Success"

# Call the function
result = hello_world()
print(f"Result: {result}")
```"""
    
    elif 'javascript' in prompt_lower or 'js' in prompt_lower:
        return """```javascript
// JavaScript example
function helloWorld() {
    console.log("Hello, World!");
    return "Success";
}

// Call the function
const result = helloWorld();
console.log(`Result: ${result}`);
```"""
    
    elif 'html' in prompt_lower:
        return """```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hello World</title>
</head>
<body>
    <h1>Hello, World!</h1>
    <p>This is a basic HTML page.</p>
</body>
</html>
```"""
    
    else:
        return f"```\n// Code example for: {prompt}\n// Please specify the programming language for better results\nconsole.log('Hello, World!');\n```"

def generate_chat_response_sync(user_message: str) -> str:
    """Generate regular chat response"""
    global llm_model, model_loaded
    
    if not model_loaded or not llm_model:
        return f"⚡ Quick Echo: {user_message}\n\n💨 (Lightning fast response!)"
    
    try:
        prompt = f"Human: {user_message}\nAssistant:"
        
        with model_lock:
            response = llm_model(
                prompt,
                max_tokens=150,
                temperature=0.7,
                top_p=0.9,
                top_k=40,
                stop=["Human:", "\nHuman:", "Assistant:", "\n\n"],
                echo=False,
                repeat_penalty=1.1,
            )
        
        if response and 'choices' in response and len(response['choices']) > 0:
            response_text = response['choices'][0]['text'].strip()
            
            if response_text:
                if len(response_text) > 1000:
                    response_text = response_text[:1000] + "..."
                
                response_text = response_text.replace("Assistant:", "").strip()
                return response_text if response_text else "🤔 Bisa dijelaskan lebih detail?"
        
        return "🤔 Hmm, bisa lebih spesifik?"
        
    except Exception as e:
        logger.error(f"Chat AI error: {e}")
        return "⚡ Sistem sedang sibuk, tapi saya tetap responsif!"

async def generate_ai_response_async(user_message: str) -> dict:
    """Async wrapper for AI generation with type detection"""
    request_type = detect_request_type(user_message)
    
    try:
        if request_type == "image":
            # Extract image prompt
            image_prompt = user_message.replace("gambar", "").replace("buatkan", "").replace("bikinin", "").strip()
            if not image_prompt:
                image_prompt = "beautiful landscape"
            
            image_data = await generate_image(image_prompt)
            return {
                "type": "image",
                "content": image_data,
                "prompt": image_prompt
            }
            
        elif request_type == "code":
            loop = asyncio.get_event_loop()
            code_response = await loop.run_in_executor(
                executor, 
                generate_code_response, 
                user_message
            )
            return {
                "type": "code",
                "content": code_response
            }
            
        else:  # chat
            loop = asyncio.get_event_loop()
            chat_response = await loop.run_in_executor(
                executor, 
                generate_chat_response_sync, 
                user_message
            )
            return {
                "type": "chat",
                "content": chat_response
            }
            
    except Exception as e:
        logger.error(f"Async AI error: {e}")
        return {
            "type": "error",
            "content": f"⚡ Quick response: Saya mengerti maksud Anda tentang '{user_message[:30]}...'"
        }

# Enhanced Command handlers
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command"""
    welcome_msg = (
        "⚡ ADVANCED AI BOT ⚡\n\n"
        "🎯 FITUR LENGKAP:\n"
        "💬 Chat AI - Respons cerdas\n"
        "🎨 Generate Gambar - Ketik 'buatkan gambar [deskripsi]'\n"
        "💻 Generate Code - Ketik 'buatkan kode [deskripsi]'\n\n"
        f"🧠 AI Status: {'🚀 Q6_K LOADED' if model_loaded else '⚡ ECHO MODE'}\n"
        f"🎨 Image: {'✅ READY' if HUGGINGFACE_TOKEN else '❌ Need HF Token'}\n\n"
        "✨ Kirim pesan untuk mulai!"
    )
    await update.message.reply_text(welcome_msg)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command"""
    help_msg = (
        "🎯 ADVANCED BOT Commands:\n\n"
        "📋 BASIC COMMANDS:\n"
        "/start - Start bot ⚡\n"
        "/help - Help menu 📖\n"
        "/status - Bot status 📊\n"
        "/speed - Speed test 🏃‍♂️\n"
        "/image [prompt] - Generate image 🎨\n"
        "/code [request] - Generate code 💻\n\n"
        "💡 USAGE EXAMPLES:\n"
        "• 'Buatkan gambar sunset di pantai'\n"
        "• 'Generate code untuk calculator python'\n"
        "• 'Jelaskan tentang AI'\n\n"
        "🚀 Smart detection - bot otomatis detect request type!"
    )
    await update.message.reply_text(help_msg)

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /status command"""
    model_info = "⚡ Q6_K TURBO MODE" if model_loaded else "💨 ECHO MODE"
    image_status = "✅ READY" if HUGGINGFACE_TOKEN else "❌ Need Token"
    
    status_msg = (
        "📊 ADVANCED BOT Status:\n\n"
        f"🤖 Bot: 🟢 RUNNING FAST\n"
        f"🧠 AI Chat: {model_info}\n"
        f"🎨 Image Gen: {image_status}\n"
        f"💻 Code Gen: ✅ READY\n"
        f"🏃‍♂️ Response: INSTANT\n"
        f"🔥 Performance: OPTIMIZED\n\n"
        "⚡ All systems operational!"
    )
    await update.message.reply_text(status_msg)

async def image_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /image command"""
    if not context.args:
        await update.message.reply_text("🎨 Usage: /image [description]\nExample: /image beautiful sunset over mountains")
        return
    
    prompt = " ".join(context.args)
    await update.message.reply_text(f"🎨 Generating image: '{prompt}'\n⏳ Please wait...")
    
    await context.bot.send_chat_action(update.effective_chat.id, "upload_photo")
    
    result = await generate_ai_response_async(f"gambar {prompt}")
    
    if result["type"] == "image" and result["content"]:
        try:
            await update.message.reply_photo(
                photo=io.BytesIO(result["content"]),
                caption=f"🎨 Generated: {result['prompt']}"
            )
        except:
            await update.message.reply_text("❌ Failed to generate image. Please try again.")
    else:
        await update.message.reply_text("❌ Image generation failed. Check HuggingFace token.")

async def code_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /code command"""
    if not context.args:
        await update.message.reply_text("💻 Usage: /code [request]\nExample: /code python function to calculate fibonacci")
        return
    
    request = " ".join(context.args)
    await update.message.reply_text(f"💻 Generating code for: '{request}'\n⏳ Please wait...")
    
    await context.bot.send_chat_action(update.effective_chat.id, "typing")
    
    result = await generate_ai_response_async(f"code {request}")
    
    if result["content"]:
        await update.message.reply_text(result["content"], parse_mode="Markdown")
    else:
        await update.message.reply_text("❌ Code generation failed. Please try again.")

async def speed_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Speed test command"""
    import time
    start_time = time.time()
    
    await context.bot.send_chat_action(update.effective_chat.id, "typing")
    
    # Test all features
    chat_result = await generate_ai_response_async("Hello test")
    end_time = time.time()
    
    speed_msg = (
        f"⚡ SPEED TEST RESULTS:\n\n"
        f"🏃‍♂️ Response time: {end_time - start_time:.2f}s\n"
        f"💬 Chat: {'✅' if chat_result else '❌'}\n"
        f"🎨 Image: {'✅' if HUGGINGFACE_TOKEN else '❌'}\n"
        f"💻 Code: ✅\n\n"
        f"🚀 Sample: {chat_result.get('content', 'N/A')[:50]}..."
    )
    
    await update.message.reply_text(speed_msg)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle messages with smart detection"""
    try:
        user_message = update.message.text
        
        # Immediate typing indicator
        asyncio.create_task(
            context.bot.send_chat_action(update.effective_chat.id, "typing")
        )
        
        # Message validation
        if len(user_message) > 2000:
            await update.message.reply_text("⚡ Pesan terlalu panjang! Maksimal 2000 karakter.")
            return
        
        # Generate response with type detection
        result = await generate_ai_response_async(user_message)
        
        if result["type"] == "image" and result["content"]:
            await context.bot.send_chat_action(update.effective_chat.id, "upload_photo")
            try:
                await update.message.reply_photo(
                    photo=io.BytesIO(result["content"]),
                    caption=f"🎨 Generated: {result['prompt']}"
                )
            except:
                await update.message.reply_text("❌ Gagal generate gambar. Coba lagi atau periksa HF token.")
                
        elif result["type"] == "code":
            await update.message.reply_text(result["content"], parse_mode="Markdown")
            
        else:  # chat or error
            await update.message.reply_text(result["content"])
        
    except Exception as e:
        logger.error(f"Message error: {e}")
        await update.message.reply_text("⚡ Oops! Tapi saya masih cepat! Coba lagi!")

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    """Enhanced error handling"""
    logger.error(f"Error: {context.error}")

def main():
    """Advanced main function"""
    print("=" * 60)
    print("🚀 ADVANCED AI TELEGRAM BOT 🚀")
    print("=" * 60)
    
    # Load model
    print("🧠 Loading AI model...")
    model_success = load_ai_model()
    
    # Setup bot
    print("⚡ Setting up advanced bot...")
    app = Application.builder().token(BOT_TOKEN).build()
    
    # Add all handlers
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("status", status_command))
    app.add_handler(CommandHandler("speed", speed_command))
    app.add_handler(CommandHandler("image", image_command))
    app.add_handler(CommandHandler("code", code_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(error_handler)
    
    print("\n" + "=" * 60)
    print("🎯 ADVANCED BOT READY! 🎯")
    print("=" * 60)
    print(f"🧠 AI Chat: {'🚀 Q6_K LOADED' if model_success else '💨 ECHO MODE'}")
    print(f"🎨 Image Gen: {'✅ READY' if HUGGINGFACE_TOKEN else '❌ Need HF Token'}")
    print(f"💻 Code Gen: ✅ READY")
    print(f"⚡ Smart Detection: ✅ ACTIVE")
    print("🛑 Press Ctrl+C to stop")
    print("=" * 60)
    
    if not HUGGINGFACE_TOKEN:
        print("💡 TIP: Set HF_TOKEN environment variable for image generation")
        print("   Get token at: https://huggingface.co/settings/tokens")
    
    try:
        app.run_polling(drop_pending_updates=True)
    except KeyboardInterrupt:
        print("\n🎯 Advanced bot stopped!")
    except Exception as e:
        print(f"❌ Bot error: {e}")
    finally:
        executor.shutdown(wait=False)

if __name__ == "__main__":
    main()
