# 🚀 MedCAFAS Local Setup Guide

Welcome to MedCAFAS! This system runs a heavy, 3-layer Medical AI hallucination detector entirely locally on your machine. Because it uses local vector databases and local AI models, patient data never leaves your computer (HIPAA-compliant by design).

Follow this step-by-step guide to get the engine running from scratch.

---

## 🛠️ Step 0: Prerequisites
Before you start, ensure you have these four free tools installed on your system:
1. **Python 3.10+** (For the AI backend engine)
2. **Node.js 18+** (For the frontend dashboard)
3. **Git** (To download this repository)
4. **[Ollama](https://ollama.com/)** (To run the Llama 3.1 LLM locally without API keys)

---

## 📥 Step 1: Download the Code and the AI
Open your terminal (Command Prompt, PowerShell, or bash) and run:

```bash
# 1. Clone the repository
git clone [https://github.com/YOUR_USERNAME/medcafas.git](https://github.com/YOUR_USERNAME/medcafas.git)

# 2. Enter the folder
cd medcafas

# 3. Download the Llama 3.1 AI model
ollama pull llama3.1

(Note: ollama pull downloads a 4.7 GB model to your machine. This might take a few minutes depending on your internet connection!)

🐍 Step 2: Set Up the Python Engine
We need to install the AI libraries (BioLinkBERT, FAISS, DeBERTa, etc.) in a safe, isolated environment so it doesn't mess with your computer's other Python projects.

Make sure you are in the root medcafas folder, then run:

Bash
# 1. Create a virtual environment named 'venv'
python -m venv venv

# 2. Activate the environment
# For Windows:
.\venv\Scripts\activate
# For Mac/Linux:
source venv/bin/activate

# 3. Install the required AI libraries
pip install -r requirements.txt
📚 Step 3: Build the Medical Library (⚠️ Do not skip!)
If you try to run the app right now, it will crash because the AI has no medical books to read! We need to compress 65,000 real medical documents into a lightning-fast vector database.

Run:

Bash
python build_kb.py
(Go grab a coffee ☕. This script downloads MedQA, PubMedQA, MedMCQA, and NIH datasets and converts them into a FAISS index. It takes about 5 to 10 minutes. You only ever have to run this once!)

⚙️ Step 4: Start the Backend Server
Once the database is built, turn on the MedCAFAS API engine:

Bash
uvicorn api:app --reload --port 8000
Leave this terminal window open! This is the brain of the operation running in the background.

💻 Step 5: Start the Frontend UI
Now we need to start the website interface.

Open a second, brand new terminal window. Navigate back into the medcafas folder, and then go into the frontend:

Bash
# 1. Enter the frontend folder
cd frontend

# 2. Install the web dependencies
npm install

# 3. Start the website
npm run dev

🎯 Step 6: The "Ah-Ha!" Moment (Test It)
Open your web browser and go to: http://localhost:3000

You will see the MedCAFAS dashboard.

Copy and paste this exact "tricky" prompt to see the system catch a fatal lie in real-time:

"What is the recommended pharmacological treatment for an acute gout attack in a patient with Stage 4 Chronic Kidney Disease (CKD)?"

Hit Analyse.

The UI will light up, the backend terminal will start printing out BioLinkBERT and DeBERTa calculations, and within a few moments, the system will flag the LLM's dangerous recommendation (NSAIDs) with a HIGH RISK warning!
