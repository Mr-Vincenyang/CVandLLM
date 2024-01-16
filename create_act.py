__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# 首先导入所需第三方库
from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from tqdm import tqdm
import os

# 获取文件路径函数
def get_files(dir_path):
    # args：dir_path，目标文件夹路径
    file_list = []
    for filepath, dirnames, filenames in os.walk(dir_path):
        # os.walk 函数将递归遍历指定文件夹
        for filename in filenames:
            # 通过后缀名判断文件类型是否满足要求
            if filename.endswith(".md"):
                # 如果满足要求，将其绝对路径加入到结果列表
                file_list.append(os.path.join(filepath, filename))
            if filename.endswith(".pdf"):
                file_list.append(os.path.join(filepath, filename))
            elif filename.endswith(".txt"):
                file_list.append(os.path.join(filepath, filename))
    # print(file_list)
    return file_list
# def get_files(dir_path, valid_extensions=(".md", ".pdf", ".txt")):
#     file_list = []
#     for root, dirs, files in os.walk(dir_path):
#         for filename in files:
#             if filename.endswith(valid_extensions):
#                 file_list.append(os.path.join(root, filename))
#     print(file_list)
#     return file_list
# 加载文件函数
def get_text(dir_path):
    # args：dir_path，目标文件夹路径
    # 首先调用上文定义的函数得到目标文件路径列表
    file_lst = get_files(dir_path)
    # docs 存放加载之后的纯文本对象
    docs = []
    # 遍历所有目标文件
    for one_file in tqdm(file_lst):
        file_type = one_file.split('.')[-1]
        if file_type == 'md':
            print("md:",one_file)
            loader = UnstructuredMarkdownLoader(one_file)
            # print("mdloader:",loader.load())
        if file_type == 'pdf':
            print("PDF ",one_file)
            try:
                loader = UnstructuredPDFLoader(one_file)
            except:
                print("PDF ERROR: ",one_file)
        elif file_type == 'txt':
            print("md:",one_file)
            loader = UnstructuredFileLoader(one_file)
        else:
            # 如果是不符合条件的文件，直接跳过
            continue
        docs.extend(loader.load())
    return docs

# 目标文件夹
tar_dir = [
    "/home/xlab-app-center/dive-into-cv-pytorch",
    "/home/xlab-app-center/so-large-lm"
]

# 加载目标文件
docs = []
for dir_path in tar_dir:
    docs.extend(get_text(dir_path))
print(docs)
# 对文本进行分块
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=150)
split_docs = text_splitter.split_documents(docs)

# 加载开源词向量模型
embeddings = HuggingFaceEmbeddings(model_name="/home/xlab-app-center/model/sentence-transformer")

# 构建向量数据库
# 定义持久化路径
persist_directory = '/home/xlab-app-center/data_base/vector_db/CVandLLM'
# 加载数据库
vectordb = Chroma.from_documents(
    documents=split_docs,
    embedding=embeddings,
    persist_directory=persist_directory  # 允许我们将persist_directory目录保存到磁盘上
)
# 将加载的向量数据库持久化到磁盘上
vectordb.persist()
