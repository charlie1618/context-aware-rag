import os

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

os.environ['GOOGLE_API_KEY'] = 'api_key'

dir = os.path.dirname(os.path.abspath(__file__))
fpath = os.path.join(dir, 'file_name.pdf')
pers_dir = os.path.join(dir, 'db')

loader = PyPDFLoader(fpath)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

docs_split = text_splitter.split_documents(docs)

embeddings = GoogleGenerativeAIEmbeddings(
    model = 'models/embedding-001'
)

db = Chroma.from_documents(
    documents = docs_split,
    embedding = embeddings,
    persist_directory = pers_dir
)

print(len(docs_split))
