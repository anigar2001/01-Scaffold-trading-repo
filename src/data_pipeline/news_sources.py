# src/data_pipeline/news_sources.py
SOURCES = [
    {
        "name": "invezz_en_bitcoin",
        "url": "https://invezz.com/news/cryptocurrency/feed/?tag=bitcoin",
        "weight": 1.0,
        "gamma": 0.8,
        "lang": "en",
    },
    {
        "name": "cointelegraph_bitcoin",
        "url": "https://cointelegraph.com/rss/tag/bitcoin",
        "weight": 0.8,
        "gamma": 0.8,
        "lang": "en",
    },
    {
        "name": "cointelegraph_followup",
        "url": "https://cointelegraph.com/rss/category/follow-up",
        "weight": 0.4,
        "gamma": 0.6,
        "lang": "en",
    },
    {
        "name": "invezz_es_bitcoin",
        "url": "https://invezz.com/es/feed/?tag=bitcoin",
        "weight": 0.5,
        "gamma": 0.8,
        "lang": "es",
    },
    {
        "name": "coindesk_all",
        "url": "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "weight": 0.6,
        "gamma": 0.8,
        "lang": "en",
    },
]
DEFAULT_SYMBOL = "BTCUSDT"
HEADLINES_CSV = "headlines_BTCUSDT.csv"     # /app/data/<este fichero>
SENTIMENT_CSV = "sentiment_BTCUSDT.csv"     # /app/data/<este fichero>
# src/data_pipeline/news_sources.py
DRIVERS_CSV   = "sentiment_drivers_BTCUSDT.csv"
