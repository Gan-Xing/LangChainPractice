from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI,VectorDBQA
from langchain.document_loaders import DirectoryLoader
from langchain.chains import RetrievalQA


import os

# 设置环境变量
os.environ["OPENAI_API_KEY"] = "sk-换个可以用的key"

# 使用环境变量
api_key = os.environ["OPENAI_API_KEY"]
# import json
# from pathlib import Path
# from pprint import pprint

# # 加载文件夹中的所有txt类型的文件
# loader = DirectoryLoader('/', glob='**/*.json')
# # 将数据转成 document 对象，每个文件会作为一个 document
# documents = loader.load()

# file_path='.chatgpt.json'
# data = json.loads(Path(file_path).read_text())
with open('./chatgpt.json') as f:
    state_of_the_union = f.read()
# 初始化加载器
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
# 切割加载的 document
documents = text_splitter.create_documents([state_of_the_union])
split_docs = text_splitter.split_documents(documents)
print(split_docs)

# 初始化 openai 的 embeddings 对象
embeddings = OpenAIEmbeddings()
# 将 document 通过 openai 的 embeddings 对象计算 embedding向量信息并临时存入 Chroma 向量数据库，用于后续匹配查询
docsearch = Chroma.from_documents(split_docs, embeddings)

# 创建问答对象
qa = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=docsearch,return_source_documents=True)
# 进行问答
result = qa({"query": "chatgpt有哪些应用"})
print(result)