import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import json

class ICHGuidelineProcessor:
    def __init__(self, persist_directory="./ich_db"):
        """初期化"""
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-large",
            model_kwargs={'device': 'cpu'}
        )

    def process_files(self, json_file_path, txt_file_path):
        """JSONとテキストファイルを処理"""
        # JSONファイルの読み込み
        with open(json_file_path, 'r', encoding='utf-8') as jf:
            metadata = json.load(jf)

        # テキストファイルの読み込み
        with open(txt_file_path, 'r', encoding='utf-8') as tf:
            content = tf.read()

        # メタデータの整理
        doc_metadata = {
            'category': metadata.get('category'),
            'code': metadata.get('code'),
            'title': metadata.get('title'),
            'source_file': metadata.get('filename')
        }

        # ドキュメントの作成
        return Document(
            page_content=content,
            metadata=doc_metadata
        )

    def split_document(self, document):
        """ドキュメントをチャンクに分割"""
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n\n", "\n", "。", "、", " "],
            length_function=len,
        )

        return splitter.split_documents([document])

    def create_vectorstore(self, documents):
        """ベクトルストアの作成"""
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        # vectorstore.persist()
        # return vectorstore