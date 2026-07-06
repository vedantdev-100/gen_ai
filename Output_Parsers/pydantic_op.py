from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate 
load_dotenv()
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

# Claude
model = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

class Person(BaseModel):
    name: str = Field(description="Name of the Person")
    age: int = Field(gt=18, description="Age of the Person")
    city: str = Field(description="Name of the city the person belongs to")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="Generate  the name, age and city of a fictional {place} person \n {format_instruction}",
    input_variables=["place"],
    partial_variables={"format_instruction":parser.get_format_instructions()}
)

# prompt = template.invoke({'place':"Indian"})
# print(prompt)
# result = model.invoke(prompt)
# final_result = parser.parse(result.content)

chain = template | model | parser

final_result = chain.invoke({'place':"American"})

print(final_result)