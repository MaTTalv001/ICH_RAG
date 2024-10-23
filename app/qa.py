import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

class ICHGuidelineQA:
    def __init__(self, persist_directory="/vectorstore/ich_db"):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-large",
            model_kwargs={'device': 'cpu'}
        )
        
        # ベクトルストアの読み込み
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )

        # ChatGPTの初期化
        self.llm = ChatOpenAI(
            temperature=0,
            model_name="gpt-4"
        )

        # プロンプトテンプレートの設定
        self.template = """あなたは医薬品規制のエキスパートです。
ICHガイドラインの内容に基づいて、質問に対して正確かつ専門的に回答してください。

参照コンテキスト:
{context}

質問: {question}

回答の際の注意点:
- ICHガイドラインの内容に基づいて回答してください
- ガイドラインの該当部分があれば、それを明示的に参照してください
- 推測が必要な場合は、その旨を明記してください
- 専門用語には必要に応じて簡潔な説明を加えてください

回答:"""

        self.prompt = ChatPromptTemplate.from_template(self.template)

        # RAGチェーンの構築
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

        self.rag_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def answer_question(self, question: str) -> str:
        """質問に対する回答を生成"""
        return self.rag_chain.invoke(question)

    def get_relevant_sources(self, question: str) -> list:
        """関連する参照元を取得"""
        docs = self.vectorstore.similarity_search(question, k=3)
        sources = []
        
        used_chunks = set()
        
        for doc in docs:
            if hasattr(doc, 'metadata'):
                chunk_preview = doc.page_content[:200].replace('\n', ' ').strip()
                
                if chunk_preview not in used_chunks:
                    used_chunks.add(chunk_preview)
                    
                    # メタデータの内容を確認
                    # print("Metadata:", doc.metadata)  # デバッグ用
                    
                    source = {
                        'title': doc.metadata.get('title', 'タイトル不明'),
                        'code': doc.metadata.get('code', 'コード不明'),
                        'category': doc.metadata.get('category', 'カテゴリ不明'),
                        'source_file': doc.metadata['source_file'] if 'source_file' in doc.metadata else None,  # 明示的に取得
                        'preview': chunk_preview
                    }
                    # print("Source:", source)  # デバッグ用
                    sources.append(source)
        
        return sources