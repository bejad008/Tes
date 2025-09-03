# pip install python-telegram-bot llama-cpp-python

import logging
import os
import sys
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Simple logging setup
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Configuration - Change this to your bot token
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8305182212:AAEDWgzi09qYZMMZDj-o-H-NX5jdvu3cm9E")

# Global variables
llm_model = None
model_loaded = False

def load_ai_model():
    """Load AI model synchronously"""
    global llm_model, model_loaded
    
    try:
        from llama_cpp import Llama
        
        print("🔄 Loading AI model...")
        
        # Try to load your preferred model
        try:
            llm_model = Llama.from_pretrained(
                repo_id="mradermacher/SEX_ROLEPLAY-3.2-1B-i1-GGUF",
                filename="SEX_ROLEPLAY-3.2-1B.i1-Q2_K.gguf",
                local_files_only=False,
                verbose=False,
                n_ctx=512,
                n_threads=2,
                n_batch=8,
                use_mlock=False,
                use_mmap=True,
                n_gpu_layers=0
            )
            
            # Test model
            test_response = llm_model("Hello", max_tokens=5)
            print("✅ AI model loaded successfully!")
            model_loaded = True
            return True
            
        except Exception as e:
            print(f"❌ Failed to load AI model: {e}")
            print("⚠️ Running in echo mode")
            model_loaded = False
            return False
            
    except ImportError:
        print("❌ llama-cpp-python not installed. Running in echo mode.")
        return False

def generate_ai_response(user_message: str) -> str:
    """Generate AI response"""
    global llm_model, model_loaded
    
    if not model_loaded or not llm_model:
        return f"🔄 Echo: {user_message}\n\n💡 (AI model tidak aktif, tapi saya tetap di sini!)"
    
    try:
        prompt = f"Human: {user_message}\nAssistant:"
        
        response = llm_model(
            prompt,
            max_tokens=100,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            stop=["Human:", "User:", "\n\n"],
            echo=False
        )
        
        if response and 'choices' in response and len(response['choices']) > 0:
            response_text = response['choices'][0]['text'].strip()
            
            if response_text:
                if len(response_text) > 800:
                    response_text = response_text[:800] + "..."
                return response_text
        
        return "Hmm, saya tidak yakin bagaimana merespons itu. Coba pertanyaan lain? 🤔"
        
    except Exception as e:
        logger.error(f"AI generation error: {e}")
        return "🤖 Maaf, sistem AI sedang bermasalah. Tapi saya tetap bisa mengobrol!"

# Command handlers
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command"""
    welcome_msg = (
        "🤖 Halo! Saya AI Bot Anda!\n\n"
        "💬 Kirim pesan apa saja dan saya akan merespons\n"
        f"🧠 Status AI: {'✅ Aktif' if model_loaded else '⚠️ Echo Mode'}\n\n"
        "🚀 Siap melayani!"
    )
    await update.message.reply_text(welcome_msg)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command"""
    help_msg = (
        "🔧 Perintah yang tersedia:\n\n"
        "/start - Mulai bot\n"
        "/help - Bantuan\n"
        "/status - Status bot\n\n"
        "💬 Atau langsung kirim pesan untuk chat dengan AI!"
    )
    await update.message.reply_text(help_msg)

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /status command"""
    status_msg = (
        "📊 Status Bot:\n\n"
        f"🤖 Bot: 🟢 Running\n"
        f"🧠 AI Model: {'✅ Loaded' if model_loaded else '❌ Not Loaded'}\n"
        f"💻 Mode: {'AI Response' if model_loaded else 'Echo Mode'}"
    )
    await update.message.reply_text(status_msg)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle text messages"""
    try:
        user_message = update.message.text
        user_name = update.effective_user.first_name or "User"
        
        logger.info(f"📩 Message from {user_name}: {user_message[:50]}...")
        
        # Show typing indicator
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action="typing"
        )
        
        # Generate response
        response_text = generate_ai_response(user_message)
        
        # Send response
        await update.message.reply_text(response_text)
        logger.info(f"📤 Response sent: {response_text[:50]}...")
        
    except Exception as e:
        logger.error(f"Message handling error: {e}")
        try:
            await update.message.reply_text(
                "😅 Oops! Ada error. Coba kirim pesan lagi ya!"
            )
        except Exception as send_error:
            logger.error(f"Failed to send error message: {send_error}")

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    """Handle errors"""
    logger.error(f"Update {update} caused error {context.error}")

def main():
    """Main function - ULTRA SIMPLE VERSION"""
    print("=" * 50)
    print("🚀 TELEGRAM AI BOT STARTING...")
    print("=" * 50)
    
    # Load AI model
    load_ai_model()
    
    # Create application
    print("🤖 Setting up Telegram bot...")
    app = Application.builder().token(BOT_TOKEN).build()
    
    # Add handlers
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("status", status_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(error_handler)
    
    print("\n" + "=" * 50)
    print("✅ BOT READY!")
    print("=" * 50)
    print(f"🧠 AI Model: {'Loaded ✅' if model_loaded else 'Echo Mode ⚠️'}")
    print("📱 Mode: Polling")
    print("🛑 Press Ctrl+C to stop")
    print("=" * 50)
    
    # Start polling - SIMPLEST METHOD
    try:
        app.run_polling(drop_pending_updates=True)
    except KeyboardInterrupt:
        print("\n👋 Bot stopped by user!")
    except Exception as e:
        print(f"❌ Bot error: {e}")

if __name__ == "__main__":
    main()