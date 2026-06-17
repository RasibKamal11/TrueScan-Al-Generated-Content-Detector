import os
import json
import asyncio
import re
from typing import Optional, Dict, Any
from loguru import logger
import google.generativeai as genai
import requests

class ExternalAIIntegrator:
    """Handles integration with free external APIs (Gemini and Hugging Face)."""
    
    def __init__(self):
        self.gemini_key = os.environ.get("GEMINI_API_KEY", "")
        self.hf_token = os.environ.get("HF_API_TOKEN", "")
        self.serper_key = os.environ.get("SERPER_API_KEY", "")
        self.groq_key = os.environ.get("GROQ_API_KEY", "")
        
        # Init Gemini
        self._gemini_ready = False
        if self.gemini_key:
            try:
                genai.configure(api_key=self.gemini_key)
                self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
                self._gemini_ready = True
                logger.success("Gemini API initialized (v1.5-flash)")
            except Exception as e:
                logger.error(f"Gemini init failed: {e}")

        # HF Config
        self.hf_models = [
            "Hello-SimpleAI/chatgpt-detector-roberta",
            "Sasha-S/roberta-large-detector",
        ]

    async def get_gemini_analysis(self, text: str) -> Optional[Dict[str, Any]]:
        """Uses Gemini to provide a deep analysis of the text's authenticity."""
        if not self._gemini_ready or not text.strip():
            return None

        prompt = f"""
        Analyze the following text for AI-generated patterns. 
        Provide a JSON response with:
        1. "ai_score": (0.0 to 1.0, where 1.0 is definitely AI)
        2. "source": (Predicted model like "GPT-4", "Claude 3", or "Human")
        3. "explanation": (A brief 2-sentence explanation of why)
        4. "red_flags": (List of specific markers like "unusually consistent sentence length", "lack of typos", "overuse of 'moreover'")

        Text to analyze:
        {text[:2000]}
        """

        try:
            # Run in thread pool since genai isn't natively async
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: self.gemini_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(response_mime_type="application/json")
            ))
            
            return json.loads(response.text)
        except Exception as e:
            logger.error(f"Gemini analysis failed: {e}")
            return None

    def google_search(self, query: str, num: int = 5) -> list[dict]:
        """Uses Serper.dev to get actual Google Search results."""
        if not self.serper_key:
            return []
        
        url = "https://google.serper.dev/search"
        headers = {
            'X-API-KEY': self.serper_key,
            'Content-Type': 'application/json'
        }
        payload = json.dumps({"q": query, "num": num})
        
        try:
            response = requests.post(url, headers=headers, data=payload, timeout=8)
            if response.status_code == 200:
                data = response.json()
                results = []
                for organic in data.get('organic', []):
                    results.append({
                        "title": organic.get('title', ''),
                        "url": organic.get('link', ''),
                        "snippet": organic.get('snippet', ''),
                        "source": "google_serper"
                    })
                return results
            return []
        except Exception as e:
            logger.error(f"Serper search failed: {e}")
            return []

    async def get_groq_prediction(self, text: str) -> Optional[float]:
        """Uses Groq (Llama 3) for ultra-fast AI detection signal."""
        if not self.groq_key or not text.strip():
            return None
        
        from groq import Groq
        client = Groq(api_key=self.groq_key)
        
        try:
            # Groq is very fast, we can use a small prompt
            prompt = f"Analyze if this text is AI-generated. Return ONLY a number between 0.0 (Human) and 1.0 (AI). Text: {text[:1000]}"
            
            loop = asyncio.get_event_loop()
            chat_completion = await loop.run_in_executor(None, lambda: client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-8b-8192",
                temperature=0.1,
            ))
            
            output = chat_completion.choices[0].message.content.strip()
            # Extract number
            match = re.search(r"(\d+\.\d+|\d+)", output)
            if match:
                return float(match.group(1))
            return None
        except Exception as e:
            logger.error(f"Groq prediction failed: {e}")
            return None

    def get_hf_inference(self, text: str) -> Optional[float]:
        """Calls Hugging Face Inference API for a second opinion."""
        if not self.hf_token or not text.strip():
            return None

        # Use the first model in the list for now
        model_id = self.hf_models[0]
        api_url = f"https://api-inference.huggingface.co/models/{model_id}"
        headers = {"Authorization": f"Bearer {self.hf_token}"}

        try:
            response = requests.post(api_url, headers=headers, json={"inputs": text[:1000]}, timeout=10)
            if response.status_code == 200:
                data = response.json()
                # Expected format: [[{"label": "LABEL_1", "score": 0.9}, ...]]
                if isinstance(data, list) and len(data) > 0:
                    results = data[0]
                    for res in results:
                        # Map labels to AI score (model dependent, usually LABEL_1 is AI)
                        if res['label'] in ['LABEL_1', 'Fake', 'AI']:
                            return res['score']
                        if res['label'] in ['LABEL_0', 'Real', 'Human']:
                            return 1.0 - res['score']
            return None
        except Exception as e:
            logger.error(f"HF Inference failed: {e}")
            return None

# Global instance for lazy loading
_integrator = None

def get_external_integrator():
    global _integrator
    if _integrator is None:
        _integrator = ExternalAIIntegrator()
    return _integrator
