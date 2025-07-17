import os
from gem_keys import key1

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma

os.environ['GOOGLE_API_KEY'] = key1

dir = os.path.dirname(os.path.abspath(__file__))
fpath = os.path.join(dir, 'merchant_of_venice.pdf')
pers_dir = os.path.join(dir, 'db')

embeddings = GoogleGenerativeAIEmbeddings(
    model = 'models/embedding-001'
)

db = Chroma(
    persist_directory = pers_dir,
    embedding_function = embeddings
)

retriever = db.as_retriever(
    search_type = 'similarity_score_threshold',
    search_kwargs = {'k':3, 'score_threshold':0.4}
)

query = input('Enter your query:\n')

relv_docs = retriever.invoke(query)

with open("test1.txt", "w") as f:
    for i, doc in enumerate(relv_docs):
        f.write(f'\n\nChunk {i + 1}:\n')
        f.write(doc.page_content)

prompt_txt = (
    'Here are some documents that might help answer the question: '
    + query
    + '\n\nRelevant Documents:\n'
    + '\n\n'.join([doc.page_content for doc in relv_docs])
    + 'Please provide an answer based on the relevant documents'
)

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

result = llm.invoke(prompt_txt)
print(f'\n{result.content}\n')