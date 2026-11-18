# Zyro Dynamics HR Help Desk

AI-powered HR policy assistant built with Streamlit, LangChain, FAISS, Hugging Face embeddings, and Groq.

## Project structure

```text
zyro-hr-helpdesk/
├── hr_helpdesk_app/
│   ├── app.py
│   ├── requirements.txt
│   └── hr_policies/
│       └── ... HR policy PDF files
├── .gitignore
└── README.md
```

## Run locally

1. Install Python 3.10 or newer.
2. Open a terminal in the `hr_helpdesk_app` folder.
3. Create and activate a virtual environment:

   **Windows PowerShell**
   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   **macOS/Linux**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Set your Groq API key.

   **Windows PowerShell (current terminal only):**
   ```powershell
   $env:GROQ_API_KEY="YOUR_GROQ_API_KEY"
   ```

   **macOS/Linux:**
   ```bash
   export GROQ_API_KEY="YOUR_GROQ_API_KEY"
   ```

6. Start the app:

   ```bash
   streamlit run app.py
   ```

Do not commit your real API key. For Streamlit Community Cloud, add `GROQ_API_KEY` under the app's Secrets settings.

## GitHub upload

Create a new GitHub repository, then from this project folder run:

```bash
git init
git add .
git commit -m "Add Zyro Dynamics HR Help Desk"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```

Replace the remote URL with your own repository URL.
