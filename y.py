# pip install llama-cpp-python huggingface-hub fastapi pyngrok python-telegram-bot uvicorn

import asyncio
import os
import time
from typing import Dict
import requests

from fastapi import FastAPI, Request
from llama_cpp import Llama
from pyngrok import ngrok
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import uvicorn
import logging
import threading

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)

# Configuration
NGROK_AUTH_TOKEN = "32BBWiXbPnsjJmq12kjOLKfT3dp_2oNJL3ntkFnRcDdbMAp5x"
BOT_TOKEN = "8305182212:AAEDWgzi09qYZMMZDj-o-H-NX5jdvu3cm9E"

# Global variables
bot: Bot = None
telegram_app: Application = None
public_url: str = "http://127.0.0.1:8000"
llm = None

# Initialize FastAPI app
app = FastAPI()

def load_model():
    """Load the model separately to handle errors"""
    global llm
    try:
        print("Loading Llama model...")
        llm = Llama.from_pretrained(
            repo_id="mradermacher/SEX_ROLEPLAY-3.2-1B-i1-GGUF",
            filename="SEX_ROLEPLAY-3.2-1B.i1-Q6_K.gguf",
            local_files_only=False,
            verbose=False,
            n_ctx=2048,
            n_threads=2
        )
        print("Model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Bot will run without AI model")
        llm = None
        return False

def setup_ngrok():
    """Setup ngrok with better error handling"""
    global public_url
    
    try:
        # Set auth token
        ngrok.set_auth_token(NGROK_AUTH_TOKEN)
        
        # Kill existing tunnels
        ngrok.kill()
        time.sleep(3)  # Wait longer for cleanup
        
        # Create new tunnel with specific configuration
        tunnel = ngrok.connect(8000, bind_tls=True, options={
            "hostname_hash": "",  # Let ngrok choose
            "inspect": False,     # Disable inspection
        })
        
        public_url = str(tunnel).replace("http://", "https://")  # Ensure https
        
        # Verify tunnel is working
        print(f"Ngrok tunnel created: {public_url}")
        
        # Test the tunnel
        try:
            response = requests.get(f"{public_url}/health", timeout=10)
            if response.status_code == 200:
                print("✅ Tunnel is working!")
            else:
                print(f"⚠️ Tunnel responded with status: {response.status_code}")
        except requests.RequestException as e:
            print(f"⚠️ Could not verify tunnel: {e}")
        
        return True
        
    except Exception as e:
        print(f"Error setting up ngrok: {e}")
        print("Will use polling mode instead of webhook")
        return False

# Telegram bot command handlers
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle the /start command"""
    await update.message.reply_text(
        "🤖 Hello! I am your AI bot. How can I assist you today?\n\n"
        "Just send me a message and I'll respond!\n"
        f"Model status: {'✅ Loaded' if llm else '❌ Not loaded (will echo messages)'}"
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle text messages"""
    try:
        user_message = update.message.text
        user_name = update.effective_user.first_name
        logger.info(f"Message from {user_name}: {user_message}")
        
        if llm is None:
            # Echo mode if model not loaded
            response_text = f"Echo: {user_message}\n\n(AI model not loaded)"
        else:
            # Generate AI response
            prompt = f"User: {user_message}\nAssistant:"
            response = llm(
                prompt,
                max_tokens=100,
                temperature=0.7,
                top_p=0.9,
                stop=["User:", "\n\n"],
                echo=False
            )
            
            # Extract the generated text
            if response and 'choices' in response and len(response['choices']) > 0:
                response_text = response['choices'][0]['text'].strip()
                if not response_text:
                    response_text = "I'm not sure how to respond to that."
            else:
                response_text = "Sorry, I couldn't generate a response."
        
        # Send response back to user
        await update.message.reply_text(response_text)
        logger.info(f"Sent response: {response_text}")
        
    except Exception as e:
        logger.error(f"Error handling message: {e}")
        await update.message.reply_text("Sorry, I encountered an error. Please try again.")

# FastAPI webhook endpoint
@app.post("/webhook")
async def telegram_webhook(request: Request):
    """Handle incoming webhook updates from Telegram"""
    try:
        json_data = await request.json()
        update = Update.de_json(json_data, bot)
        
        # Process the update
        await telegram_app.process_update(update)
        
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "Bot is running!",
        "webhook_url": f"{public_url}/webhook",
        "model_loaded": llm is not None,
        "timestamp": time.time()
    }

@app.get("/health")
async def health():
    """Health check"""
    return {"status": "ok", "timestamp": time.time()}

async def verify_bot_token():
    """Verify bot token is valid"""
    try:
        test_bot = Bot(token=BOT_TOKEN)
        bot_info = await test_bot.get_me()
        print(f"✅ Bot token verified: @{bot_info.username}")
        return True
    except Exception as e:
        print(f"❌ Bot token verification failed: {e}")
        return False

async def setup_bot_webhook():
    """Setup the Telegram bot with webhook"""
    global bot, telegram_app
    
    try:
        # Verify bot token first
        if not await verify_bot_token():
            return False
        
        # Create bot and application
        bot = Bot(token=BOT_TOKEN)
        telegram_app = Application.builder().token(BOT_TOKEN).build()
        
        # Add handlers
        telegram_app.add_handler(CommandHandler("start", start_command))
        telegram_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        
        # Initialize the application
        await telegram_app.initialize()
        
        # Prepare webhook URL
        webhook_url = f"{public_url}/webhook"
        print(f"Setting webhook to: {webhook_url}")
        
        # Delete existing webhook first
        print("Deleting existing webhook...")
        delete_result = await bot.delete_webhook(drop_pending_updates=True)
        print(f"Webhook deletion result: {delete_result}")
        
        # Wait before setting new webhook
        await asyncio.sleep(3)
        
        # Set new webhook with better parameters
        print("Setting new webhook...")
        success = await bot.set_webhook(
            url=webhook_url,
            max_connections=5,  # Increased from 1
            drop_pending_updates=True,
            allowed_updates=["message", "callback_query"]  # Specify allowed updates
        )
        
        if success:
            # Verify webhook was set
            webhook_info = await bot.get_webhook_info()
            print(f"✅ Webhook set successfully!")
            print(f"   URL: {webhook_info.url}")
            print(f"   Pending updates: {webhook_info.pending_update_count}")
            print(f"   Last error: {webhook_info.last_error_message or 'None'}")
            return True
        else:
            print("❌ Failed to set webhook")
            return False
            
    except Exception as e:
        logger.error(f"Error setting up webhook: {e}")
        # Try to get more details about the error
        try:
            webhook_info = await bot.get_webhook_info()
            print(f"Current webhook info: {webhook_info}")
        except:
            pass
        return False

async def setup_bot_polling():
    """Setup bot with polling mode (fallback)"""
    global bot, telegram_app
    
    try:
        # Verify bot token first
        if not await verify_bot_token():
            return False
        
        # Create bot and application
        bot = Bot(token=BOT_TOKEN)
        telegram_app = Application.builder().token(BOT_TOKEN).build()
        
        # Add handlers
        telegram_app.add_handler(CommandHandler("start", start_command))
        telegram_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        
        # Initialize and start polling
        await telegram_app.initialize()
        await telegram_app.start()
        
        # Make sure webhook is deleted for polling
        await bot.delete_webhook(drop_pending_updates=True)
        
        print("✅ Bot started with polling mode")
        return True
        
    except Exception as e:
        logger.error(f"Error setting up polling: {e}")
        return False

def run_fastapi():
    """Run FastAPI server in separate thread"""
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="warning")
    server = uvicorn.Server(config)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(server.serve())

async def main():
    """Main function to run the bot"""
    print("🚀 Starting Telegram Bot...")
    
    # Load model first (optional)
    print("📦 Loading AI model...")
    model_loaded = load_model()
    
    # Try to setup ngrok
    print("📡 Setting up ngrok...")
    ngrok_success = setup_ngrok()
    
    if ngrok_success:
        print("🌐 Starting FastAPI server...")
        # Start FastAPI server in background
        server_thread = threading.Thread(target=run_fastapi, daemon=True)
        server_thread.start()
        
        # Wait longer for server to be ready
        print("⏳ Waiting for server to start...")
        await asyncio.sleep(8)
        
        # Try webhook setup
        print("🔗 Setting up webhook...")
        webhook_success = await setup_bot_webhook()
        
        if webhook_success:
            print("✅ Bot running with webhook mode!")
            print(f"🌐 Public URL: {public_url}")
            print(f"🔗 Webhook: {public_url}/webhook")
            print("📱 Try sending /start to your bot!")
            
            # Keep running
            try:
                while True:
                    await asyncio.sleep(10)  # Check every 10 seconds
                    # Optionally check webhook health here
            except KeyboardInterrupt:
                print("\n🛑 Stopping bot...")
                await telegram_app.shutdown()
                ngrok.kill()
        else:
            print("❌ Webhook failed, switching to polling...")
            await setup_bot_polling()
            
            print("✅ Bot running with polling mode!")
            print("📱 Try sending /start to your bot!")
            
            # Keep polling alive
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                print("\n🛑 Stopping bot...")
                await telegram_app.stop()
                await telegram_app.shutdown()
    else:
        print("🔄 Using polling mode (no ngrok)...")
        await setup_bot_polling()
        
        print("✅ Bot running with polling mode!")
        print("📱 Try sending /start to your bot!")
        
        # Keep polling alive
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping bot...")
            await telegram_app.stop()
            await telegram_app.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
