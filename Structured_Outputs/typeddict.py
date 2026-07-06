from typing import TypedDict
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task ="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

model = ChatHuggingFace(llm=llm)

#schema
class Review(TypedDict):
    summary: str
    sentiment: str

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("This is great product with many feature and at good price, better than its competitors at the price range, but still cannot beat the contrast ratios that the  OLED panels have, since it has IPS panel, the blacks are not that constrasty")

print(result)




# class Person(TypedDict):
#     name: str
#     age: int

# new_person: Person = {'name':'vedant', 'age': 35}

# print(new_person)
