import pandas as pd
import tempfile
from datetime import datetime
import re
from typing import List, Dict
from text_unifier import TextUnifier

class ExcelProcessor:
    def __init__(self, api_key: str = "AIzaSyBhypFu2fsS07Qvi-aJuWZMHJH0YBJWpTw"):
        """
        ExcelProcessorの初期化
        """
        self.api_key = api_key
        self.text_unifier = TextUnifier(api_key)
        self.sheet_names = {
            'extraction': '抽出テキスト',
            'original': '誤った転記',
            'corrected': '正しい転記',
            'differences': '差異記録'
        }
    
    def read_excel(self, file_path):
        """
        Excelファイルを読み込み、すべてのシートのデータを返す
        """
        try:
            # まずファイル情報を取得
            xls = pd.ExcelFile(file_path)
            sheets_data = {}
            
            for sheet_name in xls.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                # NaN値を空文字に変換
                df = df.fillna('')
                sheets_data[sheet_name] = df
            
            return sheets_data
        except Exception as e:
            raise Exception(f"Excelファイルの読み込みエラー: {str(e)}")
    
    def extract_text_data(self, excel_data):
        """
        Excelデータからテキストデータを抽出
        """
        all_texts = []
        
        for sheet_name, df in excel_data.items():
            for col in df.columns:
                for idx, value in enumerate(df[col]):
                    if isinstance(value, str) and value.strip():
                        all_texts.append({
                            'sheet': sheet_name,
                            'column': col,
                            'row': idx + 2,  # Excelの行番号（ヘッダー考慮）
                            'text': value.strip()
                        })
        
        return all_texts
    
    def create_output_file(self, extraction_data, submission_data, unified_data, differences):
        """
        出力用Excelファイルを作成
        """
        try:
            # データ型の検証
            if not isinstance(unified_data, pd.DataFrame):
                raise ValueError(f"unified_dataはDataFrameである必要があります。実際の型: {type(unified_data)}")
            
            output_path = tempfile.mktemp(suffix='.xlsx')
            
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # 1枚目: 抽出するテキストのファイル
                extraction_df = self._prepare_extraction_sheet(extraction_data)
                if extraction_df.empty:
                    extraction_df = pd.DataFrame({'メッセージ': ['抽出データがありません']})
                extraction_df.to_excel(
                    writer, 
                    sheet_name=self.sheet_names['extraction'], 
                    index=False
                )
                
                # 2枚目: 誤った転記のファイル
                original_df = self._prepare_original_sheet(submission_data)
                if original_df.empty:
                    original_df = pd.DataFrame({'メッセージ': ['提出データがありません']})
                original_df.to_excel(
                    writer, 
                    sheet_name=self.sheet_names['original'], 
                    index=False
                )
                
                # 3枚目: 正しい転記のファイル
                corrected_df = self._prepare_corrected_sheet(unified_data)
                if corrected_df.empty:
                    corrected_df = pd.DataFrame({'メッセージ': ['統一データがありません']})
                corrected_df.to_excel(
                    writer, 
                    sheet_name=self.sheet_names['corrected'], 
                    index=False
                )
                
                # 4枚目: 差異記録
                if differences and len(differences) > 0:
                    differences_df = pd.DataFrame(differences)
                    differences_df.to_excel(
                        writer, 
                        sheet_name=self.sheet_names['differences'], 
                        index=False
                    )
                else:
                    # 差異がない場合も空のシートを作成
                    empty_diff_df = pd.DataFrame({'メッセージ': ['差異はありませんでした']})
                    empty_diff_df.to_excel(
                        writer, 
                        sheet_name=self.sheet_names['differences'], 
                        index=False
                    )
            
            return output_path
        except Exception as e:
            raise Exception(f"出力ファイル作成エラー: {str(e)}")
    
    def _prepare_extraction_sheet(self, extraction_data):
        """
        抽出ファイルのシートを準備
        """
        if not extraction_data or len(extraction_data) == 0:
            return pd.DataFrame()
        
        # 最初のシートを使用（複数シートがある場合）
        first_sheet = list(extraction_data.values())[0]
        return first_sheet.copy()
    
    def _prepare_original_sheet(self, submission_data):
        """
        元の提出ファイルのシートを準備
        """
        if not submission_data or len(submission_data) == 0:
            return pd.DataFrame()
        
        # 最初のシートを使用
        first_sheet = list(submission_data.values())[0]
        return first_sheet.copy()
    
    def _prepare_corrected_sheet(self, unified_data):
        """
        修正済みファイルのシートを準備
        """
        if unified_data is None or (isinstance(unified_data, pd.DataFrame) and unified_data.empty):
            return pd.DataFrame()
        
        return unified_data.copy()
    
    def normalize_text(self, text):
        """
        テキストの正規化（半角/全角、大文字/小文字の統一）
        """
        if not isinstance(text, str):
            return str(text)
        
        # 全角英数字を半角に変換
        text = text.translate(str.maketrans(
            'ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ'
            'ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ'
            '０１２３４５６７８９',
            'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            'abcdefghijklmnopqrstuvwxyz'
            '0123456789'
        ))
        
        # 全角記号を半角に変換
        text = text.translate(str.maketrans('（）［］｛｝', '()[]{}'))
        
        return text.strip()
    
    def find_similar_texts(self, target_text, text_list, threshold=0.8):
        """
        類似テキストを検索
        """
        import difflib
        
        target_normalized = self.normalize_text(target_text)
        similar_texts = []
        
        for text_info in text_list:
            text_normalized = self.normalize_text(text_info['text'])
            similarity = difflib.SequenceMatcher(
                None, 
                target_normalized.lower(), 
                text_normalized.lower()
            ).ratio()
            
            if similarity >= threshold:
                similar_texts.append({
                    **text_info,
                    'similarity': similarity,
                    'normalized_text': text_normalized
                })
        
        return sorted(similar_texts, key=lambda x: x['similarity'], reverse=True)
    
    def compare_texts(self, text1, text2):
        """
        2つのテキストを比較し、差異を検出
        """
        import difflib
        
        norm_text1 = self.normalize_text(text1)
        norm_text2 = self.normalize_text(text2)
        
        if norm_text1 == norm_text2:
            return None
        
        # 差異の詳細を取得
        diff = list(difflib.unified_diff(
            norm_text1.splitlines(keepends=True),
            norm_text2.splitlines(keepends=True),
            fromfile='元のテキスト',
            tofile='修正後のテキスト',
            lineterm=''
        ))
        
        return {
            'original': text1,
            'corrected': text2,
            'normalized_original': norm_text1,
            'normalized_corrected': norm_text2,
            'diff': ''.join(diff) if diff else None
        }
    
    def create_output_file_preserve_format(self, extraction_file_path: str, submission_file_path: str, 
                                           unified_file_path: str, differences: List[Dict]) -> str:
        """
        元のフォーマットを保持した出力用Excelファイルを作成
        """
        try:
            import openpyxl
            import tempfile
            import shutil
            
            # 出力ファイルパスを作成
            output_path = tempfile.mktemp(suffix='.xlsx')
            
            # 新しいワークブックを作成
            workbook = openpyxl.Workbook()
            
            # デフォルトシートを削除
            default_sheet = workbook.active
            workbook.remove(default_sheet)
            
            # 1枚目: 抽出ファイル（ベース）をコピー
            self._copy_excel_sheets(extraction_file_path, workbook, self.sheet_names['extraction'])
            
            # 2枚目: 提出ファイル（誤った転記）をコピー
            self._copy_excel_sheets(submission_file_path, workbook, self.sheet_names['original'])
            
            # 3枚目: 統一ファイル（正しい転記）をコピー
            self._copy_excel_sheets(unified_file_path, workbook, self.sheet_names['corrected'])
            
            # 4枚目: 差異記録を追加
            if differences and len(differences) > 0:
                self._add_differences_sheet(workbook, differences)
            else:
                # 差異がない場合も空のシートを作成
                diff_sheet = workbook.create_sheet(title=self.sheet_names['differences'])
                diff_sheet['A1'] = "差異はありませんでした"
                diff_sheet['A1'].font = openpyxl.styles.Font(bold=True)
            
            # ファイルを保存
            workbook.save(output_path)
            workbook.close()
            
            return output_path
            
        except Exception as e:
            raise Exception(f"フォーマット保持出力ファイル作成エラー: {str(e)}")
    
    def process_files(self, extraction_file_path: str, submission_file_path: str) -> tuple:
        """
        抽出ファイルと転記ファイルを処理し、統一されたファイルと差異を返す
        """
        try:
            # ファイルを読み込み
            extraction_data = self.read_excel(extraction_file_path)
            submission_data = self.read_excel(submission_file_path)
            
            # テキスト統一処理
            unified_file_path, differences = self.text_unifier.unify_texts_preserve_format(
                extraction_data, submission_data, submission_file_path
            )
            
            # 4シート出力ファイルを作成　。出力が黒になるように統一
            output_path = self.create_output_file_preserve_format(
                extraction_file_path, submission_file_path, unified_file_path, differences
            )
            self._force_black_borders(output_path)
            return output_path, differences
            
        except Exception as e:
            raise Exception(f"ファイル処理エラー: {str(e)}")
    
    def _copy_excel_sheets(self, source_file_path: str, target_workbook, new_sheet_name: str):
        """
        ソースファイルのすべてのシートを対象ワークブックにコピー
        """
        try:
            import openpyxl
            from openpyxl.utils import get_column_letter
            
            source_workbook = openpyxl.load_workbook(source_file_path)
            
            # 複数シートがある場合は最初のシートのみをコピー
            source_sheet = source_workbook.active
            target_sheet = target_workbook.create_sheet(title=new_sheet_name)
            
            # セルの値、フォーマット、スタイルをコピー
            for row in source_sheet.iter_rows():
                for cell in row:
                    target_cell = target_sheet.cell(row=cell.row, column=cell.column)
                    
                    # 値をコピー
                    target_cell.value = cell.value
                    
                    # フォントをコピー
                    if cell.font:
                        target_cell.font = openpyxl.styles.Font(
                            name=cell.font.name,
                            size=cell.font.size,
                            bold=cell.font.bold,
                            italic=cell.font.italic,
                            color=cell.font.color
                        )
                    
                    # 塗りつぶしをコピー
                    if cell.fill:
                        target_cell.fill = openpyxl.styles.PatternFill(
                            fill_type=cell.fill.fill_type,
                            start_color=cell.fill.start_color,
                            end_color=cell.fill.end_color
                        )
                    
                    # 罫線をコピー
                    if cell.border:
                        target_cell.border = openpyxl.styles.Border(
                            left=cell.border.left,
                            right=cell.border.right,
                            top=cell.border.top,
                            bottom=cell.border.bottom
                        )
                    
                    # 配置をコピー
                    if cell.alignment:
                        target_cell.alignment = openpyxl.styles.Alignment(
                            horizontal=cell.alignment.horizontal,
                            vertical=cell.alignment.vertical,
                            wrap_text=cell.alignment.wrap_text
                        )
            
            # 列幅をコピー
            for col_letter in source_sheet.column_dimensions:
                if col_letter in target_sheet.column_dimensions:
                    target_sheet.column_dimensions[col_letter].width = source_sheet.column_dimensions[col_letter].width
            
            # 行の高さをコピー
            for row_num in source_sheet.row_dimensions:
                if row_num in target_sheet.row_dimensions:
                    target_sheet.row_dimensions[row_num].height = source_sheet.row_dimensions[row_num].height
            
            source_workbook.close()
            
        except Exception as e:
            raise Exception(f"シートコピーエラー: {str(e)}")
    
    def _add_differences_sheet(self, workbook, differences: List[Dict]):
        """
        差異記録シートを追加
        """
        try:
            import openpyxl
            
            diff_sheet = workbook.create_sheet(title=self.sheet_names['differences'])
            
            # ヘッダーを設定
            headers = ['シート', '列', '行', 'セル', '元のテキスト', '修正後テキスト', '修正理由', '信頼度']
            for col, header in enumerate(headers, 1):
                cell = diff_sheet.cell(row=1, column=col)
                cell.value = header
                cell.font = openpyxl.styles.Font(bold=True)
                cell.fill = openpyxl.styles.PatternFill(
                    fill_type="solid", 
                    start_color="CCCCCC"
                )
            
            # データを追加
            for row, diff in enumerate(differences, 2):
                diff_sheet.cell(row=row, column=1).value = diff.get('シート', '')
                diff_sheet.cell(row=row, column=2).value = diff.get('列', '')
                diff_sheet.cell(row=row, column=3).value = diff.get('行', '')
                diff_sheet.cell(row=row, column=4).value = diff.get('セル', '')
                diff_sheet.cell(row=row, column=5).value = diff.get('元のテキスト', '')
                diff_sheet.cell(row=row, column=6).value = diff.get('修正後テキスト', '')
                diff_sheet.cell(row=row, column=7).value = diff.get('修正理由', '')
                diff_sheet.cell(row=row, column=8).value = diff.get('信頼度', '')
            
            # 列幅を自動調整
            for column in diff_sheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)  # 最大50文字
                diff_sheet.column_dimensions[column_letter].width = adjusted_width
            
        except Exception as e:
            raise Exception(f"差異シート追加エラー: {str(e)}") 

    def _force_black_borders(self, xlsx_path: str):
        import openpyxl
        from openpyxl.styles import Border, Side

        wb = openpyxl.load_workbook(xlsx_path)
        black = "FF000000"  # ARGBで黒
        for ws in wb.worksheets:
            for row in ws.iter_rows():
                for cell in row:
                    b = cell.border
                    if not b:
                        continue
                    def black_side(side):
                        if side is None:
                            return None
                        return Side(style=side.style, color=black)  # 線種は維持し色だけ黒
                    cell.border = Border(
                        left=black_side(b.left),
                        right=black_side(b.right),
                        top=black_side(b.top),
                        bottom=black_side(b.bottom),
                        diagonal=black_side(b.diagonal),
                        vertical=black_side(b.vertical),
                        horizontal=black_side(b.horizontal),
                    )
        wb.save(xlsx_path)
        wb.close() 