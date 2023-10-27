import openai

from src.secrets import secrets


def run_proompt(question: str, facts: list[str]) -> str:
    openai.api_key = secrets()["openai_key"]
    proompt = f"""Please answer the following question based on the given facts.
Question: "{question}"
Facts:
- {facts[0]}
- {facts[1]}
- {facts[2]}
"""

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system",
             "content": "You are a helping robot who tries to give concise answers only based on the given resources."},
            {"role": "user", "content": proompt}
        ]
    )

    return completion.choices[0].message.content
