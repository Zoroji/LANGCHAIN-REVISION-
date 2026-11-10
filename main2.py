from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

load_dotenv()

model = init_chat_model(
    model="groq:qwen/qwen3.8-27b",
    temperature=0.7
)

message = HumanMessage(
    content=[
        {
            "type": "text",
            "text": "Describe the content of this image."
        },
        {
            "type": "image_url",
            "image_url": {
                "url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRDiDG0ATZw9AYdrp4RUSy3gQSSImTYCI5pLECLyGarvw&s"
            }
        }
    ]
)

response = model.invoke([message])

print(response.content)