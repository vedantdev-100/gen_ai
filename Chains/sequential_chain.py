from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate 
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

# 1st prompt -> detailed report
prompt1 =  PromptTemplate(
    template="write a detailed report on {topic}",
    input_variables=['topic']
)

# 2nd prompt -> summary 
prompt2 =  PromptTemplate(
    template="write a 5 line summary on the following text. /n {text}",
    input_variables=['text']
)

model = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

final_result = chain.invoke({"topic":"unemployment in India"})

print(final_result)

chain.get_graph().print_ascii()