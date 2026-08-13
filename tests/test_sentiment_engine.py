import unittest

from sentiment_engine import normalize_sentiment, parse_rss


class SentimentTests(unittest.TestCase):
    def test_weighted_labels(self):
        score = normalize_sentiment(
            [
                {"label": "Bearish", "score": 0.1},
                {"label": "Neutral", "score": 0.2},
                {"label": "Bullish", "score": 0.7},
            ]
        )
        self.assertAlmostEqual(score, 0.8)

    def test_rss_is_cleaned(self):
        rss = "<rss><channel><item><title>BTC &amp; Gold</title><description><![CDATA[<b>rise</b>]]></description></item></channel></rss>"
        self.assertEqual(parse_rss(rss), ["BTC & Gold. rise"])


if __name__ == "__main__":
    unittest.main()

