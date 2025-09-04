# pip install python-telegram-bot llama-cpp-python

import logging
import os
import sys
import asyncio
import threading
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

# Global variables
llm_model = None
model_loaded = False
model_lock = threading.Lock()
executor = ThreadPoolExecutor(max_workers=3)  # For parallel AI processing

def load_ai_model():
    """Load AI model with SPEED optimizations"""
    global llm_model, model_loaded
    
    try:
        from llama_cpp import Llama
        
        print("🚀 Loading FAST AI model...")
        
        try:
            llm_model = Llama.from_pretrained(
                repo_id="mradermacher/SEX_ROLEPLAY-3.2-1B-i1-GGUF",
                filename="SEX_ROLEPLAY-3.2-1B.i1-Q2_K.gguf",  # Lightest version for speed
                local_files_only=False,
                verbose=False,
                
                # SPEED OPTIMIZATIONS
                n_ctx=256,          # Smaller context for faster processing
                n_threads=4,        # More threads for speed
                n_batch=16,         # Larger batch for efficiency
                use_mlock=True,     # Lock in memory for speed
                use_mmap=True,      # Memory mapping
                n_gpu_layers=0,     # CPU only for consistency
                
                # Additional speed settings
                seed=42,            # Fixed seed for consistency
                f16_kv=True,        # Half precision for speed
            )
            
            # Quick test
            print("⚡ Testing model speed...")
            test_response = llm_model("Hi", max_tokens=3, temperature=0.1)
            print("✅ FAST AI model loaded successfully!")
            model_loaded = True
            return True
            
        except Exception as e:
            print(f"❌ Failed to load FAST AI model: {e}")
            # Fallback to Q2_K if Q6_K fails
            try:
                print("🔄 Trying lighter model...")
                llm_model = Llama.from_pretrained(
                    repo_id="mradermacher/SEX_ROLEPLAY-3.2-1B-i1-GGUF",
                    filename="SEX_ROLEPLAY-3.2-1B.i1-Q2_K.gguf",
                    n_ctx=128,      # Even smaller for speed
                    n_threads=4,
                    n_batch=8,
                    use_mlock=True,
                    use_mmap=True,
                    verbose=False
                )
                model_loaded = True
                print("✅ Lighter model loaded!")
                return True
            except:
                print("⚠️ Running in FAST echo mode")
                model_loaded = False
                return False
            
    except ImportError:
        print("❌ llama-cpp-python not installed. Running in echo mode.")
        return False

def generate_ai_response_sync(user_message: str) -> str:
    """FAST AI response generation - synchronous"""
    global llm_model, model_loaded
    
    if not model_loaded or not llm_model:
        return f"⚡ Quick Echo: {user_message}\n\n💨 (Lightning fast response!)"
    
    try:
        # Ultra-short prompt for speed
        prompt = f"Q: {user_message}\nA:"
        
        with model_lock:  # Thread safety
            response = llm_model(
                prompt,
                max_tokens=50,      # Shorter for speed
                temperature=0.3,    # Lower for faster processing
                top_p=0.8,         # Reduced for speed
                top_k=20,          # Smaller for speed
                stop=["Q:", "Human:", "\n\n", "A:"],
                echo=False,
                repeat_penalty=1.1,  # Prevent repetition
            )
        
        if response and 'choices' in response and len(response['choices']) > 0:
            response_text = response['choices'][0]['text'].strip()
            
            if response_text:
                # Quick cleanup
                if len(response_text) > 400:  # Shorter limit for speed
                    response_text = response_text[:400] + "..."
                
                # Remove common artifacts
                response_text = response_text.replace("A:", "").strip()
                
                return response_text if response_text else "🤔 Hmm, bisa lebih spesifik?"
        
        return "🤔 Bisa dijelaskan lebih detail?"
        
    except Exception as e:
        logger.error(f"AI error: {e}")
        return "⚡ Sistem sedang sibuk, tapi saya tetap responsif!"

async def generate_ai_response_async(user_message: str) -> str:
    """Async wrapper for FAST AI generation"""
    loop = asyncio.get_event_loop()
    
    # Run AI generation in thread pool for non-blocking operation
    try:
        response = await loop.run_in_executor(
            executor, 
            generate_ai_response_sync, 
            user_message
        )
        return response
    except Exception as e:
        logger.error(f"Async AI error: {e}")
        return f"⚡ Quick response: Saya mengerti maksud Anda tentang '{user_message[:30]}...'"

# FAST Command handlers
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command - INSTANT"""
    welcome_msg = (
        "⚡ Hai! AI Bot SUPER CEPAT! ⚡\n\n"
        "💨 Respons kilat untuk semua pesan!\n"
        f"🧠 AI: {'🚀 LOADED' if model_loaded else '⚡ ECHO MODE'}\n\n"
        "✨ Kirim pesan dan rasakan kecepatannya!"
    )
    await update.message.reply_text(welcome_msg)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command - INSTANT"""
    help_msg = (
        "⚡ FAST BOT Commands:\n\n"
        "/start - Start bot ⚡\n"
        "/help - This help 📖\n"
        "/status - Bot status 📊\n"
        "/speed - Speed test 🏃‍♂️\n\n"
        "💨 Just send any message for instant AI response!"
    )
    await update.message.reply_text(help_msg)

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /status command - INSTANT"""
    status_msg = (
        "📊 SPEED BOT Status:\n\n"
        f"🤖 Bot: 🟢 RUNNING FAST\n"
        f"🧠 AI: {'⚡ TURBO MODE' if model_loaded else '💨 ECHO MODE'}\n"
        f"🏃‍♂️ Response Time: INSTANT\n"
        f"🔥 Performance: OPTIMIZED"
    )
    await update.message.reply_text(status_msg)

async def speed_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Speed test command"""
    import time
    start_time = time.time()
    
    await context.bot.send_chat_action(update.effective_chat.id, "typing")
    
    if model_loaded:
        response = await generate_ai_response_async("Hello")
        end_time = time.time()
        speed_msg = f"⚡ SPEED TEST RESULTS:\n\n🏃‍♂️ Response time: {end_time - start_time:.2f}s\n🚀 AI Response: {response[:100]}..."
    else:
        end_time = time.time()
        speed_msg = f"⚡ SPEED TEST RESULTS:\n\n💨 Echo response time: {end_time - start_time:.2f}s\n⚡ Mode: Lightning Echo"
    
    await update.message.reply_text(speed_msg)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle messages - OPTIMIZED FOR SPEED"""
    try:
        user_message = update.message.text
        
        # Immediate typing indicator (don't wait)
        asyncio.create_task(
            context.bot.send_chat_action(update.effective_chat.id, "typing")
        )
        
        # Quick message validation
        if len(user_message) > 500:  # Limit for speed
            await update.message.reply_text("⚡ Pesan terlalu panjang! Kirim yang lebih singkat untuk respons kilat.")
            return
        
        # PARALLEL processing - generate response while sending typing
        response_task = asyncio.create_task(
            generate_ai_response_async(user_message)
        )
        
        # Get response
        response_text = await response_task
        
        # Send response immediately
        await update.message.reply_text(response_text)
        
    except Exception as e:
        logger.error(f"Message error: {e}")
        # FAST error response
        await update.message.reply_text("⚡ Oops! Tapi saya masih cepat! Coba lagi!")

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    """FAST error handling"""
    logger.error(f"Error: {context.error}")

def main():
    """SPEED-OPTIMIZED main function"""
    print("=" * 50)
    print("⚡ LIGHTNING FAST TELEGRAM BOT ⚡")
    print("=" * 50)
    
    # FAST model loading
    print("🚀 Speed loading AI model...")
    model_success = load_ai_model()
    
    # Quick bot setup
    print("⚡ Lightning bot setup...")
    app = Application.builder().token(BOT_TOKEN).build()
    
    # Add handlers - minimal overhead
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("status", status_command))
    app.add_handler(CommandHandler("speed", speed_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(error_handler)
    
    print("\n" + "=" * 50)
    print("⚡ SPEED BOT READY! ⚡")
    print("=" * 50)
    print(f"🧠 AI: {'🚀 TURBO LOADED' if model_success else '💨 LIGHTNING ECHO'}")
    print("🏃‍♂️ Mode: MAXIMUM SPEED")
    print("💨 Response Time: INSTANT")
    print("🛑 Press Ctrl+C to stop")
    print("=" * 50)
    
    # FASTEST polling method
    try:
        app.run_polling(
            drop_pending_updates=True,
            pool_timeout=1,      # Faster polling
            connection_pool_size=10,  # More connections
            read_timeout=5,      # Faster timeout
            write_timeout=5,     # Faster write
        )
    except KeyboardInterrupt:
        print("\n⚡ Speed bot stopped!")
    except Exception as e:
        print(f"❌ Speed bot error: {e}")
    finally:
        executor.shutdown(wait=False)  # Quick shutdown

if __name__ == "__main__":
    main()