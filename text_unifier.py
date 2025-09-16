import google.generativeai as genai
import pandas as pd
import json
import time
from typing import Dict, List, Tuple, Any
import difflib

class TextUnifier:
    def __init__(self, api_key: str):
        """
        TextUnifierの初期化
        """
        self.api_key = api_key
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """
        Gemini APIモデルの初期化
        """
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        except Exception as e:
            raise Exception(f"Gemini API初期化エラー: {str(e)}")
    
    def unify_texts(self, extraction_data: Dict, submission_data: Dict) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        テキスト統一処理のメイン関数
        """
        try:
            # データの準備
            extraction_texts = self._extract_text_from_data(extraction_data)
            submission_df = list(submission_data.values())[0].copy()
            
            # 統一処理
            unified_df, differences = self._process_unification(
                submission_df, extraction_texts
            )
            
            return unified_df, differences
        
        except Exception as e:
            raise Exception(f"テキスト統一処理エラー: {str(e)}")
    
    def _extract_text_from_data(self, data: Dict) -> List[str]:
        """
        抽出データからテキストリストを作成
        """
        texts = []
        for sheet_name, df in data.items():
            for col in df.columns:
                for value in df[col]:
                    if isinstance(value, str) and value.strip():
                        texts.append(value.strip())
        
        return list(set(texts))  # 重複削除
    
    def _process_unification(self, submission_df: pd.DataFrame, extraction_texts: List[str]) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        統一処理の実行
        """
        unified_df = submission_df.copy()
        differences = []
        
        for col in submission_df.columns:
            for idx, value in enumerate(submission_df[col]):
                if isinstance(value, str) and value.strip():
                    # テキストの統一処理
                    unified_text, correction_info = self._unify_single_text(
                        value.strip(), extraction_texts
                    )
                    
                    if correction_info:
                        unified_df.iloc[idx, submission_df.columns.get_loc(col)] = unified_text
                        differences.append({
                            'シート': 'Sheet1',
                            '列': col,
                            '行': idx + 2,  # Excelの行番号
                            '元のテキスト': value,
                            '修正後テキスト': unified_text,
                            '修正理由': correction_info['reason'],
                            '信頼度': correction_info['confidence']
                        })
        
        return unified_df, differences
    
    def _unify_single_text(self, target_text: str, extraction_texts: List[str]) -> Tuple[str, Dict]:
        """
        単一テキストの統一処理 - AI判定最優先版 + 部分置換対応
        """
        # まず完全一致を確認
        if target_text in extraction_texts:
            return target_text, None
        
        # 意味的に同じ抽出テキストを検索
        matching_extraction = self._find_semantically_matching_extraction(target_text, extraction_texts)
        
        if matching_extraction:
            # AIで最終的な妥当性を確認
            ai_verification = self._verify_with_ai(target_text, matching_extraction['text'])
            should_correct = ai_verification.get('should_correct', False)
            
            # AI判定を最優先とする
            if should_correct:
                # 部分一致の場合は智能的な置換を実行
                if matching_extraction.get('match_type') == 'partial':
                    unified_text = self._apply_partial_replacement(target_text, matching_extraction['text'])
                else:
                    unified_text = matching_extraction['text']
                
                return unified_text, {
                    'reason': f'AI判定：{ai_verification.get("reason", "修正推奨")}（類似度: {matching_extraction["similarity"]:.2f}）',
                    'confidence': ai_verification.get('confidence', matching_extraction['confidence'])
                }
            else:
                # AI判定で修正不可となった場合は修正しない
                return target_text, None
        
        # 抽出テキストに一致しない場合は元のテキストをそのまま返す
        return target_text, None
    
    def _apply_partial_replacement(self, target_text: str, extraction_text: str) -> str:
        """
        部分一致の場合に、重要な情報を保持しながら部分置換を実行（重複防止版）
        """
        import re
        
        # 両方のテキストを正規化して比較
        target_normalized = self._deep_normalize_text(target_text)
        extraction_normalized = self._deep_normalize_text(extraction_text)
        
        # 抽出テキストがターゲットテキストの一部と一致する場合
        if extraction_normalized in target_normalized:
            # 括弧内の情報などを保持
            preserved_parts = []
            
            # 括弧内の情報を抽出（実施日、その他の情報など）
            bracket_patterns = [
                r'（[^）]*）',  # （）内の情報
                r'\([^)]*\)',   # ()内の情報
                r'【[^】]*】',   # 【】内の情報
                r'\[[^\]]*\]'   # []内の情報
            ]
            
            for pattern in bracket_patterns:
                matches = re.findall(pattern, target_text)
                preserved_parts.extend(matches)
            
            # 抽出テキストに保持すべき情報を追加（重複チェック付き）
            result = extraction_text
            for part in preserved_parts:
                # 重複防止：同じ内容や類似する内容を既に含んでいないかチェック
                should_add = True
                
                # 既に同じ括弧内容が含まれているかチェック
                if part in result:
                    should_add = False
                
                # 似たような内容（例：CR2477が既にある）をチェック
                if should_add:
                    # 括弧を除いた中身を抽出
                    inner_content = re.sub(r'[（）\(\)]', '', part)
                    if inner_content and inner_content in result:
                        should_add = False
                
                if should_add:
                    result += part
            
            return result
        
        # その他の場合は単純に抽出テキストを返す
        return extraction_text
    
    def _normalize_text(self, text: str) -> str:
        """
        テキストの正規化
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
    
    def _find_semantically_matching_extraction(self, target_text: str, extraction_texts: List[str]) -> Dict:
        """
        意味的に同じ抽出テキストを検索し、抽出テキストの正確な形式を返す - 部分一致対応強化
        """
        target_normalized = self._deep_normalize_text(target_text)
        best_match = None
        best_similarity = 0
        
        for extraction_text in extraction_texts:
            extraction_normalized = self._deep_normalize_text(extraction_text)
            
            # 完全一致の場合
            similarity = self._calculate_semantic_similarity(target_normalized, extraction_normalized)
            
            # 部分一致も確認（長いテキストの中に短いテキストが含まれる場合）
            partial_similarity = 0
            if len(extraction_normalized) < len(target_normalized):
                # 抽出テキストが短い場合、ターゲットテキストに含まれているかチェック
                if extraction_normalized in target_normalized:
                    partial_similarity = 0.95  # 部分一致の場合は高い類似度
            elif len(target_normalized) < len(extraction_normalized):
                # ターゲットテキストが短い場合、抽出テキストに含まれているかチェック
                if target_normalized in extraction_normalized:
                    partial_similarity = 0.95
            
            # より良い類似度を採用
            final_similarity = max(similarity, partial_similarity)
            
            # 専門用語の誤変換を防ぐチェック（最終類似度を使用）
            if not self._is_safe_to_convert(target_text, extraction_text, final_similarity):
                continue
            
            # デバッグ情報（開発用）
            # print(f"比較: '{target_text}' vs '{extraction_text}' -> 完全:{similarity:.3f}, 部分:{partial_similarity:.3f}, 最終:{final_similarity:.3f}")
            
            # 閾値判定（部分一致を考慮して緩和）
            threshold = 0.6  # 部分一致に対応するため閾値を下げる
            if self._contains_halfwidth_katakana(extraction_text):
                threshold = 0.5  # 半角カタカナの場合はさらに緩い閾値
            
            if final_similarity > best_similarity and final_similarity > threshold:
                best_similarity = final_similarity
                best_match = {
                    'text': extraction_text,  # 抽出テキストの元の形式を保持
                    'similarity': final_similarity,
                    'confidence': min(final_similarity + 0.1, 1.0),
                    'match_type': 'partial' if final_similarity == partial_similarity else 'complete'
                }
        
        # 完全に同じ正規化結果の場合、信頼度を最大にする
        if best_match and target_normalized == self._deep_normalize_text(best_match['text']):
            best_match['confidence'] = 1.0
            best_match['similarity'] = 1.0
            best_match['match_type'] = 'exact'
        
        return best_match
    
    def _is_safe_to_convert(self, target_text: str, extraction_text: str, similarity: float) -> bool:
        """
        変換が安全かどうかを判断（専門用語の誤変換を防ぐ） - 部分一致対応
        """
        # 文字種変更のみ（カタカナ↔ひらがな等）の場合は安全
        if self._is_character_type_change_only(target_text, extraction_text):
            return True
        
        # 部分一致（高い類似度）の場合は比較的安全とみなす
        if similarity >= 0.9:
            return True
        
        # 部分的な変更（助詞の追加削除等）の場合は安全
        if self._is_minor_modification(target_text, extraction_text):
            return True
        
        # 部分一致のパターンを確認（正規化後のテキストで）
        target_normalized = self._deep_normalize_text(target_text)
        extraction_normalized = self._deep_normalize_text(extraction_text)
        
        # 抽出テキストが対象テキストに含まれる場合（部分一致）
        if (extraction_normalized in target_normalized or 
            target_normalized in extraction_normalized):
            # 部分一致の場合、より緩い安全性チェック
            if similarity >= 0.7:  # 部分一致用の緩い閾値
                return True
        
        # 異なる語幹を持つ専門用語の変換を防ぐ（最後にチェック）
        if self._has_different_stems(target_text, extraction_text):
            return False
        
        # 中程度の類似度でも半角カタカナが含まれる場合は許可
        if self._contains_halfwidth_katakana(extraction_text) and similarity >= 0.6:
            return True
        
        return False
    
    def _is_character_type_change_only(self, text1: str, text2: str) -> bool:
        """
        文字種変更のみかどうかを判断
        """
        # 両方を正規化して比較
        norm1 = self._deep_normalize_text(text1)
        norm2 = self._deep_normalize_text(text2)
        
        return norm1 == norm2
    
    def _has_different_stems(self, text1: str, text2: str) -> bool:
        """
        異なる語幹を持つかどうかを判断
        """
        import re
        
        # 簡易的な語幹抽出（漢字部分を抽出）
        stem1 = re.findall(r'[一-龯]+', text1)
        stem2 = re.findall(r'[一-龯]+', text2)
        
        # 主要な語幹が異なる場合
        if stem1 and stem2:
            # 最も長い語幹を比較
            main_stem1 = max(stem1, key=len) if stem1 else ""
            main_stem2 = max(stem2, key=len) if stem2 else ""
            
            # 語幹が完全に異なる場合（例：「量産」vs「作業」）
            if main_stem1 and main_stem2 and main_stem1 != main_stem2:
                # 一文字以上異なる漢字がある場合は異なる語幹とみなす
                diff_chars = set(main_stem1) ^ set(main_stem2)
                return len(diff_chars) > 0
        
        return False
    
    def _is_minor_modification(self, text1: str, text2: str) -> bool:
        """
        軽微な修正（助詞の追加削除等）かどうかを判断
        """
        import re
        
        # 助詞を除去して比較
        text1_no_particles = re.sub(r'[をのはがにで]', '', text1)
        text2_no_particles = re.sub(r'[をのはがにで]', '', text2)
        
        # 助詞以外の部分が同じ場合は軽微な修正
        return self._deep_normalize_text(text1_no_particles) == self._deep_normalize_text(text2_no_particles)
    
    def _contains_halfwidth_katakana(self, text: str) -> bool:
        """
        半角カタカナが含まれているかどうかを判断
        """
        import re
        # 半角カタカナの文字範囲をチェック
        halfwidth_katakana_pattern = r'[ｱ-ﾝｧ-ｮﾞﾟ]'
        return bool(re.search(halfwidth_katakana_pattern, text))
    
    def _deep_normalize_text(self, text: str) -> str:
        """
        意味比較用の深い正規化（文字種の違いを吸収）- 半角カタカナ対応強化
        """
        if not isinstance(text, str):
            return str(text)
        
        import unicodedata
        import re
        
        # まず全角文字を半角に変換（NFKC正規化）
        text = unicodedata.normalize('NFKC', text)
        
        # 半角カタカナを全角カタカナに統一してからひらがなに変換
        text = self._normalize_katakana_variants(text)
        
        # カタカナ→ひらがな変換（比較用）
        text = self._katakana_to_hiragana(text)
        
        # 数字の統一（漢数字→算用数字）
        text = self._convert_kanji_numbers(text)
        
        # 不要な文字削除（助詞「を」「の」「は」「が」など）
        text = self._remove_optional_particles(text)
        
        # 空白・記号の正規化
        text = re.sub(r'[　\s]+', '', text)  # 空白削除
        text = re.sub(r'[・･]', '', text)    # 中点削除
        text = re.sub(r'[ー－−‐]', '', text) # 長音符削除
        
        return text.lower().strip()
    
    def _normalize_katakana_variants(self, text: str) -> str:
        """
        半角カタカナと全角カタカナを統一（半角→全角変換）
        """
        result = ""
        i = 0
        while i < len(text):
            char = text[i]
            next_char = text[i + 1] if i + 1 < len(text) else None
            
            if 'ｱ' <= char <= 'ﾝ' or char in 'ｧｨｩｪｫｯｬｭｮ':
                # 半角カタカナの処理
                if next_char == 'ﾞ':  # 濁音記号
                    # 半角カタカナ + 濁音記号 → 全角カタカナ
                    dakuten_map = {
                        'ｶ': 'ガ', 'ｷ': 'ギ', 'ｸ': 'グ', 'ｹ': 'ゲ', 'ｺ': 'ゴ',
                        'ｻ': 'ザ', 'ｼ': 'ジ', 'ｽ': 'ズ', 'ｾ': 'ゼ', 'ｿ': 'ゾ',
                        'ﾀ': 'ダ', 'ﾁ': 'ヂ', 'ﾂ': 'ヅ', 'ﾃ': 'デ', 'ﾄ': 'ド',
                        'ﾊ': 'バ', 'ﾋ': 'ビ', 'ﾌ': 'ブ', 'ﾍ': 'ベ', 'ﾎ': 'ボ',
                        'ｳ': 'ヴ'
                    }
                    result += dakuten_map.get(char, char + next_char)
                    i += 1  # 濁音記号もスキップ
                elif next_char == 'ﾟ':  # 半濁音記号
                    # 半角カタカナ + 半濁音記号 → 全角カタカナ
                    handakuten_map = {
                        'ﾊ': 'パ', 'ﾋ': 'ピ', 'ﾌ': 'プ', 'ﾍ': 'ペ', 'ﾎ': 'ポ'
                    }
                    result += handakuten_map.get(char, char + next_char)
                    i += 1  # 半濁音記号もスキップ
                else:
                    # 通常の半角カタカナ → 全角カタカナ
                    katakana_map = {
                        'ｱ': 'ア', 'ｲ': 'イ', 'ｳ': 'ウ', 'ｴ': 'エ', 'ｵ': 'オ',
                        'ｶ': 'カ', 'ｷ': 'キ', 'ｸ': 'ク', 'ｹ': 'ケ', 'ｺ': 'コ',
                        'ｻ': 'サ', 'ｼ': 'シ', 'ｽ': 'ス', 'ｾ': 'セ', 'ｿ': 'ソ',
                        'ﾀ': 'タ', 'ﾁ': 'チ', 'ﾂ': 'ツ', 'ﾃ': 'テ', 'ﾄ': 'ト',
                        'ﾅ': 'ナ', 'ﾆ': 'ニ', 'ﾇ': 'ヌ', 'ﾈ': 'ネ', 'ﾉ': 'ノ',
                        'ﾊ': 'ハ', 'ﾋ': 'ヒ', 'ﾌ': 'フ', 'ﾍ': 'ヘ', 'ﾎ': 'ホ',
                        'ﾏ': 'マ', 'ﾐ': 'ミ', 'ﾑ': 'ム', 'ﾒ': 'メ', 'ﾓ': 'モ',
                        'ﾔ': 'ヤ', 'ﾕ': 'ユ', 'ﾖ': 'ヨ',
                        'ﾗ': 'ラ', 'ﾘ': 'リ', 'ﾙ': 'ル', 'ﾚ': 'レ', 'ﾛ': 'ロ',
                        'ﾜ': 'ワ', 'ﾝ': 'ン', 'ｧ': 'ァ', 'ｨ': 'ィ', 'ｩ': 'ゥ',
                        'ｪ': 'ェ', 'ｫ': 'ォ', 'ｯ': 'ッ', 'ｬ': 'ャ', 'ｭ': 'ュ', 'ｮ': 'ョ'
                    }
                    result += katakana_map.get(char, char)
            else:
                result += char
            i += 1
        return result
    
    def _katakana_to_hiragana(self, text: str) -> str:
        """
        カタカナをひらがなに変換（比較用）- 改良版
        """
        result = ""
        for char in text:
            if 'ァ' <= char <= 'ヶ':
                # 全角カタカナ→ひらがな
                result += chr(ord(char) - ord('ア') + ord('あ'))
            else:
                result += char
        return result
    
    def _convert_kanji_numbers(self, text: str) -> str:
        """
        漢数字を算用数字に変換
        """
        kanji_num_map = {
            '零': '0', '〇': '0', '一': '1', '二': '2', '三': '3', '四': '4',
            '五': '5', '六': '6', '七': '7', '八': '8', '九': '9', '十': '10'
        }
        
        for kanji, num in kanji_num_map.items():
            text = text.replace(kanji, num)
        
        return text
    
    def _remove_optional_particles(self, text: str) -> str:
        """
        比較時に無視すべき助詞を削除
        """
        import re
        # 助詞「を」「の」「は」「が」「に」「で」などを削除
        text = re.sub(r'[をのはがにで]', '', text)
        return text
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        意味的類似度を計算 - 半角カタカナ対応強化
        """
        if not text1 or not text2:
            return 0.0
        
        # 基本的な文字列類似度
        string_similarity = difflib.SequenceMatcher(None, text1, text2).ratio()
        
        # 文字レベルの共通部分
        set1, set2 = set(text1), set(text2)
        common_chars = set1 & set2
        union_chars = set1 | set2
        char_similarity = len(common_chars) / len(union_chars) if union_chars else 0
        
        # 長さの類似度
        len1, len2 = len(text1), len(text2)
        length_similarity = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 1.0
        
        # 重み付け平均
        final_similarity = (
            string_similarity * 0.5 + 
            char_similarity * 0.3 + 
            length_similarity * 0.2
        )
        
        return final_similarity
    
    def _verify_with_ai(self, original: str, candidate: str) -> Dict:
        """
        AIによる修正の妥当性確認 - Gemini 1.5 Flash対応
        """
        try:
            prompt = f"""
以下の2つのテキストを比較し、抽出テキスト（正しい形式）に合わせた修正が適切かどうか判断してください。

現在のテキスト: "{original}"
抽出テキスト（正しい形式）: "{candidate}"

この修正システムの目的：抽出ファイルの正しい表記形式に統一すること

修正を強く推奨する基準：
1. 文字種統一（全角カタカナ→半角カタカナ、ひらがな↔カタカナ）→ 必ず推奨
2. 抽出テキストの助詞形式に合わせる調整（「を」「の」「は」「が」の追加削除）→ 必ず推奨
3. 数字形式統一（漢数字→算用数字：三→3）→ 推奨
4. 抽出テキストの表記に統一（意味内容が同じで表記方法のみ異なる）→ 推奨
5. 付帯情報（※番号、注記等）が抽出テキストで保持されている→ 推奨

修正を推奨しない基準（のみ）：
1. 異なる専門用語・概念（量産開始時≠作業開始時）→ 修正不可
2. 付帯情報が抽出テキストで失われる場合→ 修正不可

重要：抽出テキストは正しい形式として扱い、現在のテキストをその形式に合わせることが目的です。

以下のJSON形式のみで回答してください：
{{
    "should_correct": true,
    "reason": "抽出テキストの形式に統一（文字種統一＋助詞調整）",
    "confidence": 0.9
}}
"""
            
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.0,  # 一貫した判定のため温度を0に
                    max_output_tokens=150,
                    candidate_count=1
                )
            )
            
            # レスポンスをパース
            try:
                # JSONだけを抽出
                import re
                import json
                json_match = re.search(r'\{[^}]+\}', response.text)
                if json_match:
                    result = json.loads(json_match.group())
                    # 必要なフィールドを確認
                    if all(key in result for key in ['should_correct', 'reason', 'confidence']):
                        return result
                
                # パースに失敗した場合のフォールバック処理
                # 半角カタカナや文字種変更の場合は推奨
                if self._is_character_type_change_only(original, candidate):
                    return {
                        "should_correct": True,
                        "reason": "文字種統一のため（フォールバック判定）",
                        "confidence": 0.8
                    }
                
                return {
                    "should_correct": False,
                    "reason": f"AI応答解析失敗: {response.text[:50]}",
                    "confidence": 0.0
                }
                
            except (json.JSONDecodeError, AttributeError) as e:
                # エラー時のフォールバック処理
                if self._is_character_type_change_only(original, candidate):
                    return {
                        "should_correct": True,
                        "reason": "文字種統一のため（エラー時フォールバック）",
                        "confidence": 0.8
                    }
                
                return {
                    "should_correct": False,
                    "reason": f"JSON解析エラー: {str(e)}",
                    "confidence": 0.0
                }
            
        except Exception as e:
            # API呼び出しエラー時のフォールバック処理
            if self._is_character_type_change_only(original, candidate):
                return {
                    "should_correct": True,
                    "reason": "文字種統一のため（API エラー時フォールバック）",
                    "confidence": 0.8
                }
            
            return {
                "should_correct": False,
                "reason": f"API呼び出しエラー: {str(e)}",
                "confidence": 0.0
            }
    

    
    def unify_texts_preserve_format(self, extraction_data: Dict, submission_data: Dict, submission_file_path: str) -> Tuple[str, List[Dict]]:
        """
        元のフォーマットを保持したテキスト統一処理
        """
        try:
            import openpyxl
            from openpyxl.utils import get_column_letter
            import tempfile
            import shutil
            
            # 抽出データからテキストリストを作成
            extraction_texts = self._extract_text_from_data(extraction_data)
            
            # 提出ファイルをコピーして編集用に準備
            output_path = tempfile.mktemp(suffix='.xlsx')
            shutil.copy2(submission_file_path, output_path)
            
            # openpyxlでファイルを開く
            workbook = openpyxl.load_workbook(output_path)
            differences = []
            
            # 各シートを処理
            for sheet_name in workbook.sheetnames:
                worksheet = workbook[sheet_name]
                
                # セル単位で処理
                for row in worksheet.iter_rows():
                    for cell in row:
                        if cell.value and isinstance(cell.value, str) and cell.value.strip():
                            original_text = cell.value.strip()
                            
                            # テキストの統一処理
                            unified_text, correction_info = self._unify_single_text(
                                original_text, extraction_texts
                            )
                            
                            # 修正が必要な場合
                            if correction_info:
                                cell.value = unified_text
                                differences.append({
                                    'シート': sheet_name,
                                    '列': get_column_letter(cell.column),
                                    '行': cell.row,
                                    'セル': f"{get_column_letter(cell.column)}{cell.row}",
                                    '元のテキスト': original_text,
                                    '修正後テキスト': unified_text,
                                    '修正理由': correction_info['reason'],
                                    '信頼度': correction_info['confidence']
                                })
            
            # 修正済みファイルを保存
            workbook.save(output_path)
            workbook.close()
            
            return output_path, differences
            
        except Exception as e:
            raise Exception(f"フォーマット保持統一処理エラー: {str(e)}")
    
    def create_correction_summary(self, differences: List[Dict]) -> str:
        """
        修正サマリーの作成
        """
        if not differences:
            return "修正対象のテキストはありませんでした。"
        
        summary = f"合計 {len(differences)} 件の修正を実行しました。\n\n"
        
        # 修正理由別の集計
        reason_counts = {}
        for diff in differences:
            reason = diff['修正理由'].split(',')[0]  # 最初の理由のみ取得
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        
        summary += "修正理由別の件数:\n"
        for reason, count in reason_counts.items():
            summary += f"- {reason}: {count}件\n"
        
        return summary 