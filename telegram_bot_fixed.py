# pip install llama-cpp-python huggingface-hub fastapi pyngrok python-telegram-bot uvicorn

import asyncio
import os
from typing import Dict

from fastapi import FastAPI, Request
from llama_cpp import Llama
from pyngrok import ngrok
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import uvicorn
import logging

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load the model
print("Loading Llama model...")
llm = Llama.from_pretrained(
    repo_id="mradermacher/SEX_ROLEPLAY-3.2-1B-i1-GGUF",
    filename="SEX_ROLEPLAY-3.2-1B.i1-Q6_K.gguf",
    local_files_only=False,
    verbose=False
)
print("Model loaded successfully!")

# Initialize FastAPI app
app = FastAPI()

# Configuration
NGROK_AUTH_TOKEN = "32BBWiXbPnsjJmq12kjOLKfT3dp_2oNJL3ntkFnRcDdbMAp5x"
BOT_TOKEN = "8305182212:AAEDWgzi09qYZMMZDj-o-H-NX5jdvu3cm9E"

# Global variables
bot: Bot = None
telegram_app: Application = None
public_url: str = "http://127.0.0.1:8000"

# Set up ngrok
def setup_ngrok():
    global public_url
    ngrok.set_auth_token(NGROK_AUTH_TOKEN)
    
    # Close any existing ngrok connections
    ngrok.kill()
    
    try:
        # Use the correct format for pyngrok
        public_url = ngrok.connect(8000, bind_tls=True)
        print(f"Public URL: {public_url}")
        return public_url
    except Exception as e:
        print(f"Error setting up ngrok: {e}")
        print("Attempting to use localhost as fallback.")
        public_url = "http://127.0.0.1:8000"
        return public_url

# Telegram bot command handlers
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle the /start command"""
    await update.message.reply_text(
        "Hello! I am your AI bot. How can I assist you today?\n"
        "Just send me a message and I'll respond using AI!"
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle text messages"""
    try:
        user_message = update.message.text
        logger.info(f"Received message: {user_message}")
        
        # Generate response using Llama model
        prompt = f"<|begin_of_text|>{user_message}<|end_of_text|>"
        response = llm(
            prompt,
            max_tokens=150,
            temperature=0.7,
            top_p=0.9,
            stop=["<|end_of_text|>", "\n\n"]
        )
        
        # Extract the generated text
        generated_text = response['choices'][0]['text'].strip()
        
        # Send response back to user
        await update.message.reply_text(generated_text)
        logger.info(f"Sent response: {generated_text}")
        
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
    return {"status": "Bot is running!", "webhook_url": f"{public_url}/webhook"}

async def setup_bot():
    """Setup the Telegram bot"""
    global bot, telegram_app
    
    # Create bot and application
    bot = Bot(token=BOT_TOKEN)
    telegram_app = Application.builder().token(BOT_TOKEN).build()
    
    # Add handlers
    telegram_app.add_handler(CommandHandler("start", start_command))
    telegram_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Initialize the application
    await telegram_app.initialize()
    
    # Set webhook
    webhook_url = f"{public_url}/webhook"
    await bot.set_webhook(url=webhook_url)
    logger.info(f"Webhook set to: {webhook_url}")

async def main():
    """Main function to run the bot"""
    # Setup ngrok
    setup_ngrok()
    
    # Setup bot
    await setup_bot()
    
    # Start FastAPI server
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)
    
    print(f"Bot is running!")
    print(f"Public URL: {public_url}")
    print(f"Webhook URL: {public_url}/webhook")
    
    try:
        await server.serve()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        await telegram_app.shutdown()
        ngrok.disconnect(public_url)
        ngrok.kill()

if __name__ == "__main__":
    asyncio.run(main())