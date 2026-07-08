from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
import os

os.environ["USER_AGENT"] = "MyLangChainApp/1.0"

load_dotenv()

# Model 1 - Claude
model = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

prompt = PromptTemplate(
    template='Answer the following question \n {question} from the following text - \n {text}',
    input_variables=['question','text']
)

parser = StrOutputParser()

url = 'https://www.amazon.in/Lenovo-i5-13450HX-300Nits-Graphics-83DV007GIN/dp/B0D49W5KZP/ref=pd_rhf_se_s_pd_sbs_rvi_d_sccl_2_2/524-4271563-8211700?psc=1'
loader = WebBaseLoader(url)

docs = loader.load()
# print(len(docs))
# print(docs[0].page_content)

chain = prompt | model | parser

# print(chain.invoke({'question':'What is the prodcut that we are talking about?', 'text':docs[0].page_content}))
print(chain.invoke({'question':'Explain the display features of this produt?', 'text':docs[0].page_content}))