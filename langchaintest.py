import os

# 设置环境变量
os.environ["OPENAI_API_KEY"] = "sk-换个可以用的key"

# 使用环境变量
api_key = os.environ["OPENAI_API_KEY"]

from langchain.llms import OpenAI

llm = OpenAI(model_name="gpt-3.5-turbo",max_tokens=1024)
print(llm("怎么学习AI"))