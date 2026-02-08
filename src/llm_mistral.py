import os
from mistralai import Mistral

DEFAULT_MODEL = "mistral-large-latest"  # tu peux changer ensuite

class MistralLLM:
    def __init__(self, model: str = DEFAULT_MODEL):
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise RuntimeError("MISTRAL_API_KEY manquant. Fais: export MISTRAL_API_KEY='...'")
        self.model = model
        self.client = Mistral(api_key=api_key)

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.3) -> str:
        # Chat completions via SDK officiel
        res = self.client.chat.complete(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )
        return res.choices[0].message.content
