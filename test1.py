from google import genai

client = genai.Client()

response = client.models.embed_content(
    model="gemini-embedding-2",
    contents="Hello world"
)

print(len(response.embeddings[0].values))