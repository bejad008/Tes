# pip install python-telegram-bot llama-cpp-python

import logging
import os
import sys
import asyncio
import threading
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes

# Logging setup
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Configuration
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_BOT_TOKEN_HERE")
WORKSPACE_DIR = Path("bot_workspace")
WORKSPACE_DIR.mkdir(exist_ok=True)

# Global variables
llm_model = None
model_loaded = False
model_lock = threading.Lock()
executor = ThreadPoolExecutor(max_workers=3)

def load_qwen_model():
    """Load Qwen model optimized for coding"""
    global llm_model, model_loaded
    
    try:
        from llama_cpp import Llama
        
        print("🚀 Loading Qwen Coding Model...")
        
        try:
            llm_model = Llama.from_pretrained(
                repo_id="mradermacher/Qwen3-4B-Claude-Sonnet-4-Reasoning-Distill-Safetensor-GGUF",
                filename="Qwen3-4B-Claude-Sonnet-4-Reasoning-Distill-Safetensor-Q4_K_M.gguf",
                local_files_only=False,
                verbose=False,
                
                # Optimized for Python coding
                n_ctx=4096,         # Large context for code files
                n_threads=4,        
                n_batch=512,        # Good for code generation
                use_mlock=True,     
                use_mmap=True,      
                n_gpu_layers=0,
                
                seed=42,
                f16_kv=True,
            )
            
            # Test model
            print("⚡ Testing Qwen model...")
            test_response = llm_model("def hello():", max_tokens=10, temperature=0.1)
            print("✅ Qwen Coding Model loaded successfully!")
            model_loaded = True
            return True
            
        except Exception as e:
            print(f"❌ Failed to load Qwen model: {e}")
            print("⚡ Running in fallback mode")
            model_loaded = False
            return False
            
    except ImportError:
        print("❌ llama-cpp-python not installed. Install with: pip install llama-cpp-python")
        return False

def get_main_keyboard():
    """Get main menu inline keyboard"""
    keyboard = [
        [
            InlineKeyboardButton("📝 Buat File Python", callback_data="create_file"),
            InlineKeyboardButton("📂 Lihat File", callback_data="list_files")
        ],
        [
            InlineKeyboardButton("✏️ Edit File", callback_data="edit_file"),
            InlineKeyboardButton("🔍 Baca File", callback_data="read_file")
        ],
        [
            InlineKeyboardButton("🚀 Generate Code", callback_data="generate_code"),
            InlineKeyboardButton("🗑️ Hapus File", callback_data="delete_file")
        ],
        [
            InlineKeyboardButton("ℹ️ Help", callback_data="help"),
            InlineKeyboardButton("📊 Status", callback_data="status")
        ]
    ]
    return InlineKeyboardMarkup(keyboard)

def get_file_list_keyboard(files):
    """Get keyboard for file selection"""
    keyboard = []
    for file in files:
        keyboard.append([InlineKeyboardButton(f"📄 {file.name}", callback_data=f"select_file:{file.name}")])
    
    keyboard.append([InlineKeyboardButton("🔙 Kembali", callback_data="main_menu")])
    return InlineKeyboardMarkup(keyboard)

def list_python_files():
    """Get list of Python files in workspace"""
    return list(WORKSPACE_DIR.glob("*.py"))

def read_file_content(filename):
    """Read Python file content"""
    try:
        file_path = WORKSPACE_DIR / filename
        if file_path.exists() and file_path.suffix == '.py':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        return None
    except Exception as e:
        logger.error(f"Error reading file {filename}: {e}")
        return None

def write_file_content(filename, content):
    """Write content to Python file"""
    try:
        if not filename.endswith('.py'):
            filename += '.py'
        
        file_path = WORKSPACE_DIR / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    except Exception as e:
        logger.error(f"Error writing file {filename}: {e}")
        return False

def delete_file(filename):
    """Delete Python file"""
    try:
        file_path = WORKSPACE_DIR / filename
        if file_path.exists() and file_path.suffix == '.py':
            file_path.unlink()
            return True
        return False
    except Exception as e:
        logger.error(f"Error deleting file {filename}: {e}")
        return False

def generate_python_code(prompt):
    """Generate Python code using Qwen model"""
    global llm_model, model_loaded
    
    if not model_loaded or not llm_model:
        return generate_fallback_python_code(prompt)
    
    try:
        # Specialized Python coding prompt
        coding_prompt = f"""You are an expert Python programmer. Write clean, well-documented Python code for the following request.

Request: {prompt}

Python code:
```python"""
        
        with model_lock:
            response = llm_model(
                coding_prompt,
                max_tokens=800,
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
                # Clean up the code
                code_text = code_text.replace('```python', '').replace('```', '').strip()
                return code_text
        
        return generate_fallback_python_code(prompt)
        
    except Exception as e:
        logger.error(f"Code generation error: {e}")
        return generate_fallback_python_code(prompt)

def generate_fallback_python_code(prompt):
    """Generate simple Python code when AI model fails"""
    prompt_lower = prompt.lower()
    
    if 'calculator' in prompt_lower or 'kalkulator' in prompt_lower:
        return '''def calculator():
    """Simple calculator function"""
    while True:
        try:
            print("\\nSimple Calculator")
            print("1. Addition (+)")
            print("2. Subtraction (-)")
            print("3. Multiplication (*)")
            print("4. Division (/)")
            print("5. Exit")
            
            choice = input("Choose operation (1-5): ")
            
            if choice == '5':
                print("Thank you for using calculator!")
                break
            
            if choice in ['1', '2', '3', '4']:
                num1 = float(input("Enter first number: "))
                num2 = float(input("Enter second number: "))
                
                if choice == '1':
                    result = num1 + num2
                    print(f"{num1} + {num2} = {result}")
                elif choice == '2':
                    result = num1 - num2
                    print(f"{num1} - {num2} = {result}")
                elif choice == '3':
                    result = num1 * num2
                    print(f"{num1} * {num2} = {result}")
                elif choice == '4':
                    if num2 != 0:
                        result = num1 / num2
                        print(f"{num1} / {num2} = {result}")
                    else:
                        print("Error: Division by zero!")
            else:
                print("Invalid choice!")
                
        except ValueError:
            print("Error: Please enter valid numbers!")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    calculator()'''
    
    elif 'web scraper' in prompt_lower or 'scraping' in prompt_lower:
        return '''import requests
from bs4 import BeautifulSoup
import csv

def web_scraper(url, output_file="scraped_data.csv"):
    """Simple web scraper"""
    try:
        # Send GET request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract data (customize based on needs)
        data = []
        
        # Example: Extract all links
        links = soup.find_all('a', href=True)
        for link in links:
            data.append({
                'text': link.get_text().strip(),
                'url': link['href']
            })
        
        # Save to CSV
        if data:
            with open(output_file, 'w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=['text', 'url'])
                writer.writeheader()
                writer.writerows(data)
            
            print(f"Scraped {len(data)} items to {output_file}")
        else:
            print("No data found!")
            
    except requests.RequestException as e:
        print(f"Request error: {e}")
    except Exception as e:
        print(f"Error: {e}")

# Usage
if __name__ == "__main__":
    url = input("Enter URL to scrape: ")
    web_scraper(url)'''
    
    else:
        return f'''# Python code for: {prompt}

def main():
    """Main function for {prompt}"""
    print("Hello, World!")
    # Add your code implementation here
    pass

if __name__ == "__main__":
    main()'''

# Command Handlers
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command"""
    welcome_msg = (
        "🐍 **PYTHON CODING BOT** 🐍\n\n"
        "🎯 **FITUR UTAMA:**\n"
        "• 📝 Buat file Python baru\n"
        "• ✏️ Edit file Python yang ada\n"
        "• 🔍 Baca dan tampilkan isi file\n"
        "• 🚀 Generate kode Python dengan AI\n"
        "• 📂 Kelola file dalam workspace\n\n"
        f"🤖 AI Status: {'🚀 Qwen Model Ready' if model_loaded else '⚡ Fallback Mode'}\n"
        f"📁 Workspace: `{WORKSPACE_DIR.absolute()}`\n\n"
        "✨ Pilih aksi di bawah untuk memulai!"
    )
    
    await update.message.reply_text(
        welcome_msg, 
        parse_mode="Markdown",
        reply_markup=get_main_keyboard()
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command"""
    help_msg = (
        "📚 **PYTHON CODING BOT HELP** 📚\n\n"
        "🎮 **COMMANDS:**\n"
        "/start - Mulai bot dan tampilkan menu utama\n"
        "/help - Tampilkan bantuan ini\n"
        "/files - Lihat semua file Python\n"
        "/workspace - Info lokasi workspace\n\n"
        "🎯 **CARA PENGGUNAAN:**\n"
        "1️⃣ Gunakan inline buttons untuk navigasi\n"
        "2️⃣ Untuk membuat file: pilih 'Buat File Python'\n"
        "3️⃣ Untuk generate code: pilih 'Generate Code'\n"
        "4️⃣ Semua file disimpan dalam workspace bot\n\n"
        "💡 **TIPS:**\n"
        "• Gunakan nama file yang jelas (tanpa .py)\n"
        "• Bot akan otomatis menambahkan ekstensi .py\n"
        "• Kode yang dihasilkan AI siap dijalankan\n\n"
        "❓ Butuh bantuan lebih lanjut? Tanya saja!"
    )
    
    await update.message.reply_text(
        help_msg,
        parse_mode="Markdown"
    )

async def files_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /files command"""
    files = list_python_files()
    
    if files:
        file_list = "\n".join([f"📄 `{f.name}`" for f in files])
        msg = f"📂 **File Python dalam Workspace:**\n\n{file_list}\n\n📁 Lokasi: `{WORKSPACE_DIR.absolute()}`"
    else:
        msg = "📂 **Workspace kosong**\n\nBelum ada file Python. Buat file baru dengan tombol di bawah!"
    
    await update.message.reply_text(
        msg,
        parse_mode="Markdown",
        reply_markup=get_main_keyboard()
    )

async def workspace_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /workspace command"""
    files = list_python_files()
    file_count = len(files)
    
    workspace_info = (
        f"📁 **WORKSPACE INFO** 📁\n\n"
        f"📍 **Lokasi:** `{WORKSPACE_DIR.absolute()}`\n"
        f"📊 **Total File:** {file_count} file Python\n"
        f"💾 **Status:** {'Tersedia' if WORKSPACE_DIR.exists() else 'Error'}\n\n"
        f"🤖 **AI Model:** {'🟢 Qwen Active' if model_loaded else '🟡 Fallback Mode'}\n\n"
    )
    
    if files:
        workspace_info += "**File List:**\n"
        for i, f in enumerate(files[:10], 1):  # Show max 10 files
            size_kb = f.stat().st_size / 1024
            workspace_info += f"{i}. `{f.name}` ({size_kb:.1f} KB)\n"
        
        if len(files) > 10:
            workspace_info += f"... dan {len(files) - 10} file lainnya"
    
    await update.message.reply_text(
        workspace_info,
        parse_mode="Markdown",
        reply_markup=get_main_keyboard()
    )

# Callback Query Handler
async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle inline button callbacks"""
    query = update.callback_query
    await query.answer()
    
    data = query.data
    
    if data == "main_menu":
        await query.edit_message_text(
            "🐍 **PYTHON CODING BOT** 🐍\n\nPilih aksi yang ingin dilakukan:",
            parse_mode="Markdown",
            reply_markup=get_main_keyboard()
        )
    
    elif data == "create_file":
        context.user_data['action'] = 'create_file'
        await query.edit_message_text(
            "📝 **BUAT FILE PYTHON BARU**\n\n"
            "Kirim nama file yang ingin dibuat (tanpa .py)\n"
            "Contoh: `calculator` atau `web_scraper`\n\n"
            "❌ Ketik /cancel untuk membatalkan",
            parse_mode="Markdown"
        )
    
    elif data == "list_files":
        files = list_python_files()
        if files:
            await query.edit_message_text(
                "📂 **DAFTAR FILE PYTHON**\n\nPilih file untuk melihat detail:",
                parse_mode="Markdown",
                reply_markup=get_file_list_keyboard(files)
            )
        else:
            await query.edit_message_text(
                "📂 **Workspace Kosong**\n\n"
                "Belum ada file Python. Buat file baru terlebih dahulu!",
                parse_mode="Markdown",
                reply_markup=InlineKeyboardMarkup([[
                    InlineKeyboardButton("📝 Buat File Baru", callback_data="create_file"),
                    InlineKeyboardButton("🔙 Menu Utama", callback_data="main_menu")
                ]])
            )
    
    elif data == "read_file":
        files = list_python_files()
        if files:
            context.user_data['action'] = 'read_file'
            await query.edit_message_text(
                "🔍 **BACA FILE PYTHON**\n\nPilih file yang ingin dibaca:",
                parse_mode="Markdown",
                reply_markup=get_file_list_keyboard(files)
            )
        else:
            await query.edit_message_text(
                "📂 **Tidak Ada File**\n\nBelum ada file Python untuk dibaca.",
                parse_mode="Markdown",
                reply_markup=get_main_keyboard()
            )
    
    elif data == "edit_file":
        files = list_python_files()
        if files:
            context.user_data['action'] = 'edit_file'
            await query.edit_message_text(
                "✏️ **EDIT FILE PYTHON**\n\nPilih file yang ingin diedit:",
                parse_mode="Markdown",
                reply_markup=get_file_list_keyboard(files)
            )
        else:
            await query.edit_message_text(
                "📂 **Tidak Ada File**\n\nBelum ada file Python untuk diedit.",
                parse_mode="Markdown",
                reply_markup=get_main_keyboard()
            )
    
    elif data == "delete_file":
        files = list_python_files()
        if files:
            context.user_data['action'] = 'delete_file'
            await query.edit_message_text(
                "🗑️ **HAPUS FILE PYTHON**\n\n⚠️ **PERHATIAN:** File yang dihapus tidak dapat dikembalikan!\n\nPilih file yang ingin dihapus:",
                parse_mode="Markdown",
                reply_markup=get_file_list_keyboard(files)
            )
        else:
            await query.edit_message_text(
                "📂 **Tidak Ada File**\n\nBelum ada file Python untuk dihapus.",
                parse_mode="Markdown",
                reply_markup=get_main_keyboard()
            )
    
    elif data == "generate_code":
        context.user_data['action'] = 'generate_code'
        await query.edit_message_text(
            "🚀 **GENERATE PYTHON CODE**\n\n"
            "Deskripsikan kode Python yang ingin dibuat:\n\n"
            "💡 **Contoh:**\n"
            "• `Buat calculator sederhana`\n"
            "• `Web scraper untuk mengambil data`\n"
            "• `Program untuk sorting data`\n\n"
            "❌ Ketik /cancel untuk membatalkan",
            parse_mode="Markdown"
        )
    
    elif data == "status":
        files = list_python_files()
        status_msg = (
            f"📊 **STATUS BOT** 📊\n\n"
            f"🤖 **Bot:** 🟢 Aktif\n"
            f"🧠 **AI Model:** {'🟢 Qwen Ready' if model_loaded else '🟡 Fallback Mode'}\n"
            f"📁 **Workspace:** {len(files)} file Python\n"
            f"💾 **Storage:** `{WORKSPACE_DIR.absolute()}`\n\n"
            f"⚡ **Performance:** Optimal\n"
            f"🐍 **Python Version:** {sys.version.split()[0]}"
        )
        
        await query.edit_message_text(
            status_msg,
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup([[
                InlineKeyboardButton("🔙 Menu Utama", callback_data="main_menu")
            ]])
        )
    
    elif data == "help":
        help_msg = (
            "📚 **BANTUAN CEPAT** 📚\n\n"
            "🎯 **FITUR UTAMA:**\n"
            "• 📝 Buat file Python baru\n"
            "• ✏️ Edit file yang sudah ada\n"
            "• 🔍 Baca isi file\n"
            "• 🚀 Generate kode dengan AI\n"
            "• 🗑️ Hapus file\n\n"
            "💡 **TIPS PENGGUNAAN:**\n"
            "• Gunakan nama file yang jelas\n"
            "• Bot otomatis tambah ekstensi .py\n"
            "• Kode AI siap dijalankan\n\n"
            "❓ Ketik /help untuk bantuan lengkap"
        )
        
        await query.edit_message_text(
            help_msg,
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup([[
                InlineKeyboardButton("🔙 Menu Utama", callback_data="main_menu")
            ]])
        )
    
    elif data.startswith("select_file:"):
        filename = data.split(":", 1)[1]
        action = context.user_data.get('action')
        
        if action == 'read_file':
            content = read_file_content(filename)
            if content:
                # Split long content
                if len(content) > 3000:
                    content_preview = content[:3000] + "\n\n... (truncated)"
                else:
                    content_preview = content
                
                msg = f"🔍 **FILE: {filename}**\n\n```python\n{content_preview}\n```"
                
                keyboard = [
                    [InlineKeyboardButton("✏️ Edit File", callback_data=f"edit_selected:{filename}")],
                    [InlineKeyboardButton("🗑️ Hapus File", callback_data=f"delete_selected:{filename}")],
                    [InlineKeyboardButton("🔙 Daftar File", callback_data="list_files")]
                ]
                
                await query.edit_message_text(
                    msg,
                    parse_mode="Markdown",
                    reply_markup=InlineKeyboardMarkup(keyboard)
                )
            else:
                await query.edit_message_text(
                    f"❌ Error membaca file `{filename}`",
                    parse_mode="Markdown",
                    reply_markup=get_main_keyboard()
                )
        
        elif action == 'edit_file':
            context.user_data['edit_filename'] = filename
            content = read_file_content(filename)
            
            if content:
                preview = content[:500] + "..." if len(content) > 500 else content
                msg = (
                    f"✏️ **EDIT FILE: {filename}**\n\n"
                    f"**Isi saat ini:**\n```python\n{preview}\n```\n\n"
                    f"Kirim kode Python baru untuk mengganti seluruh isi file:\n\n"
                    f"❌ Ketik /cancel untuk membatalkan"
                )
            else:
                msg = f"✏️ **EDIT FILE: {filename}**\n\nKirim kode Python baru:"
            
            await query.edit_message_text(msg, parse_mode="Markdown")
        
        elif action == 'delete_file':
            keyboard = [
                [InlineKeyboardButton("✅ Ya, Hapus!", callback_data=f"confirm_delete:{filename}")],
                [InlineKeyboardButton("❌ Batalkan", callback_data="list_files")]
            ]
            
            await query.edit_message_text(
                f"🗑️ **KONFIRMASI HAPUS**\n\n"
                f"Yakin ingin menghapus file `{filename}`?\n\n"
                f"⚠️ **File yang dihapus tidak dapat dikembalikan!**",
                parse_mode="Markdown",
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
    
    elif data.startswith("confirm_delete:"):
        filename = data.split(":", 1)[1]
        if delete_file(filename):
            await query.edit_message_text(
                f"✅ **FILE DIHAPUS**\n\nFile `{filename}` berhasil dihapus dari workspace.",
                parse_mode="Markdown",
                reply_markup=get_main_keyboard()
            )
        else:
            await query.edit_message_text(
                f"❌ **ERROR**\n\nGagal menghapus file `{filename}`.",
                parse_mode="Markdown",
                reply_markup=get_main_keyboard()
            )
    
    elif data.startswith("edit_selected:"):
        filename = data.split(":", 1)[1]
        context.user_data['action'] = 'edit_file'
        context.user_data['edit_filename'] = filename
        
        await query.edit_message_text(
            f"✏️ **EDIT FILE: {filename}**\n\n"
            f"Kirim kode Python baru untuk mengganti seluruh isi file:\n\n"
            f"❌ Ketik /cancel untuk membatalkan",
            parse_mode="Markdown"
        )
    
    elif data.startswith("delete_selected:"):
        filename = data.split(":", 1)[1]
        keyboard = [
            [InlineKeyboardButton("✅ Ya, Hapus!", callback_data=f"confirm_delete:{filename}")],
            [InlineKeyboardButton("❌ Batalkan", callback_data=f"select_file:{filename}")]
        ]
        
        await query.edit_message_text(
            f"🗑️ **KONFIRMASI HAPUS**\n\n"
            f"Yakin ingin menghapus file `{filename}`?\n\n"
            f"⚠️ **File yang dihapus tidak dapat dikembalikan!**",
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

# Message Handler
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle text messages based on current action"""
    user_message = update.message.text
    action = context.user_data.get('action')
    
    if user_message == '/cancel':
        context.user_data.clear()
        await update.message.reply_text(
            "❌ **Aksi dibatalkan**\n\nKembali ke menu utama:",
            parse_mode="Markdown",
            reply_markup=get_main_keyboard()
        )
        return
    
    if action == 'create_file':
        filename = user_message.strip()
        if not filename:
            await update.message.reply_text("❌ Nama file tidak boleh kosong!")
            return
        
        # Generate default Python template
        default_content = f'''#!/usr/bin/env python3
"""
{filename}.py - Created by Python Coding Bot
Description: Add your description here
"""

def main():
    """Main function"""
    print("Hello from {filename}!")
    # Add your code here
    pass

if __name__ == "__main__":
    main()
'''
        
        if write_file_content(filename, default_content):
            keyboard = [
                [InlineKeyboardButton("✏️ Edit File", callback_data=f"edit_selected:{filename}.py")],
                [InlineKeyboardButton("🚀 Generate Code", callback_data="generate_code")],
                [InlineKeyboardButton("🔙 Menu Utama", callback_data="main_menu")]
            ]
            
            await update.message.reply_text(
                f"✅ **FILE DIBUAT**\n\n"
                f"File `{filename}.py` berhasil dibuat dengan template dasar.\n\n"
                f"📁 Lokasi: `{WORKSPACE_DIR / (filename + '.py')}`",
                parse_mode="Markdown",
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
        else:
            await update.message.reply_text(
                f"❌ **ERROR**\n\nGagal membuat file `{filename}.py`",
                parse_mode="Markdown",
                reply_markup=get_main_keyboard()
            )
        
        context.user_data.clear()
    
    elif action == 'edit_file':
        filename = context.user_data.get('edit_filename')
        if filename:
            if write_file_content(filename.replace('.py', ''), user_message):
                keyboard = [
                    [InlineKeyboardButton("🔍 Baca File", callback_data=f"select_file:{filename}")],
                    [InlineKeyboardButton("🔙 Menu Utama", callback_data="main_menu")]
                ]
                
                await update.message.reply_text(
                    f"✅ **FILE DIPERBARUI**\n\n"
                    f"File `{filename}` berhasil diperbarui dengan kode baru.",
                    parse_mode="Markdown",
                    reply_markup=InlineKeyboardMarkup(keyboard)
                )
            else:
                await update.message.reply_text(
                    f"