from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate 
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv
import os

load_dotenv()

def word_counter(text):
    return len(text.split())

# runnable_word_counter = RunnableLambda(word_counter)
# print(runnable_word_counter.invoke('hi there How are you ?'))

# Model 1 - Claude
model = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=['topic']
)

joke_generator_chain = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    # 'word_count': RunnableLambda(lambda x: len(x.split()))
    'word_count': RunnableLambda(word_counter)
})

final_chain = RunnableSequence(joke_generator_chain, parallel_chain)

result = final_chain.invoke({'topic': 'cricket'})

final_result = """"{} \n word count - {}""".format(result['joke'], result['word_count'])
print(final_result)