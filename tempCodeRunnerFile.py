import PIL.Image

model = genai.GenerativeModel("gemini-1.5-flash")
organ = PIL.Image.open(media / "download.jpeg")
response = model.generate_content(["Tell me about this text", organ])
print(response.text)