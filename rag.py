import 

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma

os.environ['GOOGLE_API_KEY'] = 'api_key'

dir = os.path.dirname(os.path.abspath(__file__))
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
