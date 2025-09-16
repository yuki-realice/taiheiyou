import streamlit as st
import pandas as pd
import tempfile
import os
from datetime import datetime
from excel_processor import ExcelProcessor

from dotenv import load_dotenv

# .env（任意）やシステム環境変数を読み込む
load_dotenv()
API_KEY = os.getenv("EXCEL_PROC_API_KEY", "")  # ← ここに一本化

def main():
    st.title("Excel テキスト統一システム")
    st.write("抽出ファイルの形式に合わせて転記ファイルのテキストを統一します")

    # 先にAPIキーをチェック（未設定ならアプリ内で案内して停止）
    if not API_KEY:
        st.error("APIキーが設定されていません。環境変数 EXCEL_PROC_API_KEY を設定してから再実行してください。")
        st.info("例: ターミナルで\n  echo 'export EXCEL_PROC_API_KEY=\"<YOUR_KEY>\"' >> ~/.zshrc\n  source ~/.zshrc")
        return

    # ファイルアップロード
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("抽出ファイル（正しい形式）")
        extraction_file = st.file_uploader(
            "抽出ファイルをアップロード",
            type=['xlsx', 'xls'],
            key="extraction"
        )
    with col2:
        st.subheader("転記ファイル（修正対象）")
        submission_file = st.file_uploader(
            "転記ファイルをアップロード",
            type=['xlsx', 'xls'],
            key="submission"
        )

    if extraction_file and submission_file:
        st.write("---")

        if st.button("テキスト統一処理を実行", type="primary"):
            try:
                # プログレス
                progress_bar = st.progress(0)
                status_text = st.empty()

                status_text.text("ファイルを読み込み中...")
                progress_bar.progress(20)

                # 一時保存
                with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_ext:
                    tmp_ext.write(extraction_file.getvalue())
                    extraction_path = tmp_ext.name
                with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_sub:
                    tmp_sub.write(submission_file.getvalue())
                    submission_path = tmp_sub.name

                # 実処理
                status_text.text("テキスト統一処理中...")
                progress_bar.progress(50)

                processor = ExcelProcessor(API_KEY)  # ← ここで統一したAPI_KEYを使用
                result_file, differences = processor.process_files(extraction_path, submission_path)

                progress_bar.progress(80)
                status_text.text("結果ファイル準備中...")

                # 結果表示
                progress_bar.progress(100)
                status_text.text("処理完了！")
                st.success(f"処理が完了しました！ {len(differences)} 件の修正を実行しました。")

                if differences:
                    st.subheader("修正内容の詳細")
                    df_differences = pd.DataFrame(differences)
                    st.dataframe(df_differences, use_container_width=True)

                    st.subheader("主な修正内容")
                    for i, diff in enumerate(differences[:5], 1):
                        with st.expander(f"修正 {i}: {diff.get('セル','(不明)')}"):
                            c1, c2 = st.columns(2)
                            with c1:
                                st.write("**修正前:**")
                                st.code(diff.get('元のテキスト', ''))
                            with c2:
                                st.write("**修正後:**")
                                st.code(diff.get('修正後テキスト', ''))
                            st.write(f"**理由:** {diff.get('修正理由','')}")
                            conf = diff.get('信頼度', None)
                            if conf is not None:
                                try:
                                    st.write(f"**信頼度:** {float(conf):.3f}")
                                except Exception:
                                    st.write(f"**信頼度:** {conf}")
                else:
                    st.info("修正対象のテキストはありませんでした。")

                # ダウンロード
                with open(result_file, 'rb') as f:
                    file_data = f.read()
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                download_name = f"統一済みファイル_{timestamp}.xlsx"
                st.download_button(
                    label="📥 統一済みファイルをダウンロード",
                    data=file_data,
                    file_name=download_name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary"
                )

                # ファイル情報
                st.write("**ダウンロードファイルの構成:**")
                st.write("- シート1: 抽出テキスト（正しい形式）")
                st.write("- シート2: 元の転記ファイル")
                st.write("- シート3: 修正済み転記ファイル")
                st.write("- シート4: 修正内容の詳細記録")

                # プレビュー
                st.subheader("修正済みファイルの内容確認")
                try:
                    result_df = pd.read_excel(result_file, sheet_name="修正済み転記")
                    st.write("**修正済みシートの内容:**")
                    st.dataframe(result_df, use_container_width=True)
                except Exception as e:
                    st.warning(f"ファイル内容の表示でエラーが発生しました: {str(e)}")

                # 片付け
                try:
                    os.unlink(extraction_path)
                    os.unlink(submission_path)
                except Exception:
                    pass

            except Exception as e:
                st.error(f"処理中にエラーが発生しました: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    else:
        st.info("両方のファイルをアップロードしてください。")

    # 使い方
    with st.expander("📖 使用方法"):
        st.write("""
        **このシステムの機能:**
        1. 抽出ファイル（正しい形式）と転記ファイル（修正対象）をアップロード
        2. AIが文字種統一、助詞調整、数字形式統一などを自動実行
        3. 元のExcelフォーマット（セルの色、サイズ等）を完全保持
        4. 異なる専門用語の誤変換を防止

        **修正される内容例:**
        - 全角カタカナ → 半角カタカナ（ダストトレイ → ﾀﾞｽﾄﾄﾚｲ）
        - 助詞の統一（砂を廃棄 → 砂廃棄）
        - 数字形式統一（三日 → 3日）
        - 文字種統一（ホコリ → ほこり）

        **修正されない内容:**
        - 異なる専門用語（量産開始時 ≠ 作業開始時）
        - 付帯情報の削除（※2などの注記は保持）
        """)

if __name__ == "__main__":
    main()
