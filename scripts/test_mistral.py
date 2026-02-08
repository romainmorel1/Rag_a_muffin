import os
from mistralai import Mistral

client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
res = client.chat.complete(
    model="mistral-large-latest",
    messages=[
        {"role": "system", "content": "Tu es Chef Muffin. Réponds en français."},
        {"role": "user", "content": "Propose un muffin au chocolat en 3 phrases."},
    ],
    temperature=0.3,
)

print(res.choices[0].message.content)
