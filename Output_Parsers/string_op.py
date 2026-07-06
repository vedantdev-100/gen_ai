from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate 
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# HuggingFace
# llm = HuggingFaceEndpoint(
#     repo_id="Qwen/Qwen2.5-7B-Instruct",
#     task ="text-generation",
#     huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
# )
# model = ChatHuggingFace(llm=llm) 

# Claude
model = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

# 1st prompt -> detailed report
template1 =  PromptTemplate(
    template="write a detailed report on {topic}",
    input_variables=['topic']
)

# 2nd prompt -> summary 
template2 =  PromptTemplate(
    template="write a 5 line summary on the following text. /n {text}",
    input_variables=['text']
)

# prompt1 = template1.invoke({'topic':'black hole'})
# result = model.invoke(prompt1)

# prompt2 = template2.invoke({'text':result.content})
# result1 = model.invoke(prompt2)

# print(result1.content)

#####  String structured output
parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic':'black hole'})

print(result)