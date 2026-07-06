from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate 
load_dotenv()
from langchain_anthropic import ChatAnthropic
# from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StructuredOutputParser, ResponseSchema   

load_dotenv()

# Claude
model = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

schema = [
    ResponseSchema(name='fact_1', description='Fact 1 about the topic'),
    ResponseSchema(name='fact_2', description='Fact 2 about the topic'),
    ResponseSchema(name='fact_3', description='Fact 3 about the topic'),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    # template='Give me the name, age and city of a fictional person \n {format_instruction},',
    template='Give me 3 facts about {topic} \n {format_instruction},',
    input_variables=[],  
    partial_variables={'format_instruction': parser.get_format_instructions()}
)


prompt = template.invoke({'topic': 'black hole'})

result = model.invoke(prompt)

final_result = parser.parse(result.content)

print(final_result)