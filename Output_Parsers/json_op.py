from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate 
load_dotenv()
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

# Claude
model = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

parser = JsonOutputParser()

template = PromptTemplate(
    # template='Give me the name, age and city of a fictional person \n {format_instruction},',
    template='Give me 5 facts about {topic} \n {format_instruction},',
    input_variables=[],  
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

# prompt = template.format()
# result = model.invoke(prompt)

# final_result = parser.parse(result.content)

chain = template | model | parser
result = chain.invoke({'topic': 'black hole'})

print(result)
print(type(result))