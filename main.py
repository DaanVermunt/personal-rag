from src.lib import search_str
from src.proompting import run_proompt

question = "Is it allowed to work for a competitor"

res = search_str(question)
responses = [hit.entity.text.replace('\n', "") for hit in res[0]]

response = run_proompt(question, responses)
print(response)
