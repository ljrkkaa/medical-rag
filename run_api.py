import os

import uvicorn
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

if __name__ == "__main__":
    # API keys can be set in .env file or environment variables
    # Example .env file:
    # DASHSCOPE_API_KEY=sk-...
    # OPENAI_API_KEY=sk-...
    # API_HOST=0.0.0.0
    # API_PORT=8000
    uvicorn.run(
        "MedicalRag.api.app:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8005")),
        workers=1,  # MUST 1: session state lives in process memory
        log_level="info",
        reload=False,
    )
