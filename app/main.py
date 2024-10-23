import streamlit as st
from processor import ICHGuidelineProcessor
from qa import ICHGuidelineQA
import os
import glob
import pandas as pd

st.title("ICH Guidelines QA System")

# モード選択
mode = st.sidebar.selectbox(
    "モードを選択してください",
    ["RAG (Q&A)", "ベクトル化", "データセット"]
)

if mode == "ベクトル化":
    st.header("ICHガイドラインのベクトル化")
    
    if st.button("ベクトル化を実行"):
        try:
            # ディレクトリの存在確認
            data_dir = "/data/ich_guidelines"
            if not os.path.exists(data_dir):
                st.error(f"データディレクトリが見つかりません: {data_dir}")
                st.stop()

            # ファイルの存在確認
            json_files = glob.glob(os.path.join(data_dir, "*.json"))
            if not json_files:
                st.error(f"JSONファイルが見つかりません: {data_dir}")
                st.stop()

            processor = ICHGuidelineProcessor(persist_directory="/vectorstore/ich_db")
            all_chunks = []

            with st.spinner("ファイルを処理中..."):
                for json_file in json_files:
                    base_name = os.path.splitext(json_file)[0]
                    txt_file = base_name + '.txt'
                    
                    if os.path.exists(txt_file):
                        # ドキュメントの処理
                        doc = processor.process_files(json_file, txt_file)
                        # ドキュメントの分割
                        chunks = processor.split_document(doc)
                        all_chunks.extend(chunks)
                        st.write(f"処理完了: {os.path.basename(json_file)}")

            if all_chunks:
                with st.spinner("ベクトルストアを作成中..."):
                    vectorstore = processor.create_vectorstore(all_chunks)
                st.success(f"ベクトル化が完了しました。{len(all_chunks)}チャンクを処理しました。")
            else:
                st.warning("処理可能なファイルが見つかりませんでした")

        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")

elif mode == "RAG (Q&A)" :  # RAGモード
    st.header("ICHガイドラインQ&A")
    
    try:
        qa_system = ICHGuidelineQA(persist_directory="/vectorstore/ich_db")
        
        question = st.text_input("質問を入力してください")
        
        if question:
            with st.spinner("回答を生成中..."):
                answer = qa_system.answer_question(question)
                sources = qa_system.get_relevant_sources(question)
            
            st.write("回答:")
            st.write(answer)
            
            st.write("参照ソース:")
            for source in sources:
                st.write(f"- {source['title']} ({source['code']})")
    except Exception as e:
        st.error(f"エラーが発生しました: {str(e)}")

else: #データセット確認
    st.header("収録データセットの確認")
    
    df = pd.read_csv("/data/dataset/ich.csv") 
    st.dataframe(df)
