import openai
from starlette.config import Config

config = Config(".env")
OPEN_AI_KEY = config('OPENAI_API_KEY')

def summarize_text_openai(text):
    openai.api_key = OPEN_AI_KEY
    response = openai.Completion.create(
    model="text-davinci-003",
    prompt=text,
    temperature=0.7,
    max_tokens=64,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0
    )
    #print(response)
    return response