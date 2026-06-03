import unittest
from zumen_indexer.parser import (
    normalize_spaces,
    clean_title,
    infer_discipline,
    chunk_text,
    score_text_quality,
)


class TestZumenIndexer(unittest.TestCase):

    def test_normalize_spaces(self):
        self.assertEqual(normalize_spaces("  A01   1階平面図   "), "A01 1階平面図")
        self.assertEqual(normalize_spaces("\t図面名\t\t平面図\t"), "図面名 平面図")

    def test_clean_title(self):
        self.assertEqual(clean_title("図面リスト ●"), "図面リスト")
        self.assertEqual(clean_title("配置図 -"), "配置図")
        self.assertEqual(clean_title(" 求積図・面積表 ・ "), "求積図・面積表")

    def test_infer_discipline(self):
        self.assertEqual(infer_discipline("A05", "1階平面図"), "architecture")
        self.assertEqual(infer_discipline("C10", "基礎伏図"), "structure")
        self.assertEqual(infer_discipline("M02", "空調設備図"), "mechanical")
        self.assertEqual(infer_discipline("E01", "電灯設備"), "electrical")
        self.assertEqual(infer_discipline("X99", "基礎断面図"), "structure")
        self.assertEqual(infer_discipline("XYZ", "その他"), "unknown")

    def test_chunk_text(self):
        sample_text = "line1\nline2\nline3\n"
        # max_chars=11 allows "line1\nline2" (len is 11)
        chunks = chunk_text(sample_text, max_chars=11)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0], "line1\nline2")
        self.assertEqual(chunks[1], "line3")

    def test_score_text_quality(self):
        # Empty
        self.assertEqual(score_text_quality(""), 0.0)
        
        # High quality (contains Japanese and alphanumeric words)
        jp_text = "本日は晴天なり。1階平面図と構造図を作成します。"
        score = score_text_quality(jp_text)
        self.assertTrue(score > 0.3)


if __name__ == "__main__":
    unittest.main()
