import os
import requests
import json
from pdfminer.high_level import extract_text
import streamlit as st

class ICHDownloader:
    def __init__(self, output_dir="/data/ich_guidelines/"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def download_pdf(self, url, filename):
        response = requests.get(url)
        if response.status_code == 200:
            with open(filename, 'wb') as f:
                f.write(response.content)
            return True
        return False

    def pdf_to_text(self, pdf_path):
        return extract_text(pdf_path)

    def create_metadata(self, row, text):
        return {
            "category": row["Category"],
            "code": row["Code"],
            "title": row["Title"],
            "step": row["Step"],
            "filename": row["Filename"],
            "content_summary": text[:500]
        }

    def process_guideline(self, row, progress_text):
        url = f"https://www.pmda.go.jp/files/{row['Filename']}"
        pdf_filename = os.path.join(self.output_dir, row["Filename"])
        txt_filename = os.path.join(self.output_dir, f"{row['Code']}.txt")
        json_filename = os.path.join(self.output_dir, f"{row['Code']}.json")

        if os.path.exists(txt_filename) and os.path.exists(json_filename):
            st.info(f"既に存在します: {row['Code']}")
            return

        progress_text.text(f"処理中: {row['Filename']}")

        if not self.download_pdf(url, pdf_filename):
            st.error(f"ダウンロード失敗: {row['Filename']}")
            return

        try:
            text = self.pdf_to_text(pdf_filename)

            with open(txt_filename, 'w', encoding='utf-8') as f:
                f.write(text)

            metadata = self.create_metadata(row, text)
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            st.success(f"処理完了: {row['Filename']}")

        except Exception as e:
            st.error(f"エラー発生 {row['Filename']}: {str(e)}")

        finally:
            if os.path.exists(pdf_filename):
                os.remove(pdf_filename)

    def process_all(self, df):
        progress_text = st.empty()
        total_files = len(df)
        progress_bar = st.progress(0)

        for index, row in df.iterrows():
            self.process_guideline(row, progress_text)
            progress = (index + 1) / total_files
            progress_bar.progress(progress)

        st.success("全てのファイルの処理が完了しました")