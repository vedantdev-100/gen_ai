from langchain_community.document_loaders import TextLoader
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv
import os

load_dotenv()

loader = TextLoader('cricket.txt', encoding='utf-8')
docs = loader.load()

# Model 1 - Claude
model = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

parser = StrOutputParser()

prompt = PromptTemplate(
    template="Write a summary about a {poem}",
    input_variables=['poem']
)

chain = prompt | model | parser
result = chain.invoke({'poem':docs[0].page_content})
print(result)










# print(type(docs))
# print(len(docs))
# print((docs[0]))