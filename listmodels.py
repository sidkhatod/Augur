from google import genai

client = genai.Client()

for model in client.models.list():
    print(model.name)

