from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
import os

load_dotenv()
OPENAI_API_KEY=os.environ["OPENAI_API_KEY"]

ChatGPT = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    temperature=0,
    model_name="gpt-3.5-turbo"
)

prompt = [
    HumanMessage(content="list three colors"),
]

chat = ChatGPT(prompt)

print(chat.content)
print("*" * 80)

from langchain.schema import SystemMessage, HumanMessage

messages = [
    SystemMessage(content="you list items in reverse alphabetical order"),
    HumanMessage(content="list three colors"),
]

chat = ChatGPT(messages)

print(chat.content)
print("*" * 80)

# Imports
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# System Message
system_template="You are a helpful assistant that translates {input_language} to English Boston Accent."
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

# Human Message
human_template="{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

# Prompt
prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

# Chat
chat = ChatGPT(prompt.format_prompt(input_language="English", text="Do you want to visit the Harbor or Harvard Yard").to_messages())

# Print
print(chat.content)
print("*" * 80)

from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI

loader = TextLoader("../data/us-constitution.txt")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
txt_search = Chroma.from_documents(texts, embeddings)

ChatGPT = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    temperature=0,
    model_name="gpt-3.5-turbo"
)

chatTXT = RetrievalQA.from_chain_type(llm=ChatGPT, chain_type="stuff", retriever=txt_search.as_retriever())

prompt = "What is the 2nd ammendment"
chat = chatTXT.run(prompt)

print("*" * 80)
print(chat)
print("*" * 80)

from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("../data/pkd-metz.pdf")
pages = loader.load_and_split()

embeddings = OpenAIEmbeddings()
pdf_search = Chroma.from_documents(pages, embeddings, persist_directory="db/")

from langchain.chat_models import ChatOpenAI

ChatGPT = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    temperature=0,
    model_name="gpt-4"
)

chatPDF = RetrievalQA.from_chain_type(llm=ChatGPT, chain_type="stuff", retriever=pdf_search.as_retriever())

prompt = "are we living in a computer generated simulation"
chat = chatPDF.run(prompt)

print("*" * 80)
print(chat)
print("*" * 80)
