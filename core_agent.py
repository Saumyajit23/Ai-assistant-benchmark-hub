import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class UnifiedAssistant:
    def __init__(self):
        # Initialize Frontier Client via Groq Cloud
        self.frontier_client = OpenAI(
            base_url=os.getenv("FRONTIER_BASE_URL", "https://api.groq.com/openai/v1"),
            api_key=os.getenv("FRONTIER_API_KEY")
        ) if os.getenv("FRONTIER_API_KEY") else None
        
        # Initialize OSS Client via Hugging Face Serverless API
        self.oss_client = OpenAI(
            base_url=os.getenv("OSS_BASE_URL", "https://router.huggingface.co/v1"),
            api_key=os.getenv("OSS_API_KEY")
        ) if os.getenv("OSS_API_KEY") else None

    def get_response(self, model_type: str, prompt: str, chat_history: list) -> str:
        """
        Processes multi-turn conversations using the chosen backend[cite: 20, 21].
        chat_history format: [{"role": "user"/"assistant", "content": "text"}]
        """
        # Format conversation history cleanly for OpenAI standard formatting
        messages = [{"role": m["role"], "content": m["content"]} for m in chat_history]
        messages.append({"role": "user", "content": prompt})

        # Normalize the string to lowercase to eliminate any casing or spacing issues
        normalized_type = str(model_type).lower()

        # Ultra-forgiving string matching
        if any(keyword in normalized_type for keyword in ["frontier", "groq", "llama", "gemini"]):
            return self._call_frontier(messages)
        elif any(keyword in normalized_type for keyword in ["open", "source", "oss"]):
            return self._call_oss(messages)
        else:
            raise ValueError(f"Unknown model type selected: {model_type}")

    def _call_frontier(self, messages: list) -> str:
        if not self.frontier_client:
            return "❌ Frontier API Key/Config missing. Check your .env file."
        
        model_name = os.getenv("FRONTIER_MODEL_NAME", "llama-3.3-70b-versatile")
        
        try:
            response = self.frontier_client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"⚠️ Frontier API Error: {str(e)}"

    def _call_oss(self, messages: list) -> str:
        if not self.oss_client:
            return "❌ Hugging Face Token missing. Check your .env file."
        
        model_name = os.getenv("OSS_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
        
        try:
            response = self.oss_client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"⚠️ OSS Router Error: {str(e)}"