import streamlit as st
from processor import ICHGuidelineProcessor
from qa import ICHGuidelineQA
import os
import glob
import pandas as pd
from ich_downloader import ICHDownloader


# グローバルなインスタンス保持
@st.cache_resource
def initialize_qa_system():
    return ICHGuidelineQA(persist_directory="/vectorstore/ich_db")

# セッション状態の効率的な管理
if "qa_system" not in st.session_state:
    st.session_state.qa_system = initialize_qa_system()

st.title("ICH Guidelines Q&A System")

# サイドバーでモード選択
mode = st.sidebar.radio(
    "モード選択",
    ["ユーザーモード", "収録ガイドライン表示", "管理者モード"],
    index=0  # デフォルトはユーザーモード
)

if mode == "ユーザーモード":
    st.header("ICHガイドラインQ&A")
    
    # RAG機能
    try:
        qa_system = ICHGuidelineQA(persist_directory="/vectorstore/ich_db")
        
        question = st.text_input("質問を入力してください")
        
        if question:
            with st.spinner("回答を生成中..."):
                answer = qa_system.answer_question(question)
                sources = qa_system.get_relevant_sources(question)
            
            st.markdown("### 💡 回答")
            st.markdown(f">{answer}")
            
            st.markdown("### 📚 参照ソース")
            tabs = st.tabs([f"ソース {i+1}" for i in range(len(sources))])
            
            for tab, source in zip(tabs, sources):
                with tab:
                    st.markdown(f"**ガイドライン:** {source['title']} ({source['code']})")
                    filename = source.get('source_file')
                    if filename:
                        url = f"https://www.pmda.go.jp/files/{filename}"
                        st.markdown(f"[元のPDFを開く]({url}) 📄")
                    st.markdown(f"**カテゴリ:** {source['category']}")
                    st.markdown("**関連箇所:**")
                    st.text_area(
                        label="",
                        value=source['preview'],
                        height=300,
                        disabled=False
                    )
    except Exception as e:
        st.error(f"エラーが発生しました: {str(e)}")

elif mode == "収録ガイドライン表示":
    st.header("収録ガイドライン一覧")
    df = pd.read_csv("/data/dataset/ich.csv")
    st.dataframe(df)

else:  # 管理者モード
    st.warning("⚠️ 管理者モードです。システムの設定変更が可能です。")
    
    admin_mode = st.radio(
        "管理機能を選択",
        ["ベクトル化", "データセット管理"]
    )

    if admin_mode == "ベクトル化":
        st.header("ICHガイドラインのベクトル化")
        if st.button("ベクトル化を実行"):
            try:
                data_dir = "/data/ich_guidelines"
                if not os.path.exists(data_dir):
                    st.error(f"データディレクトリが見つかりません: {data_dir}")
                    st.stop()

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
                            doc = processor.process_files(json_file, txt_file)
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

    else:  # データセット管理
        st.header("データセット管理")
        
        df = pd.read_csv("/data/dataset/ich.csv")
        st.dataframe(df)
        
        if st.button("ガイドラインをダウンロード"):
            try:
                downloader = ICHDownloader()
                downloader.process_all(df)
            except Exception as e:
                st.error(f"エラーが発生しました: {str(e)}")