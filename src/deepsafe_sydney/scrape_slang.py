import json
import logging
import time
from pathlib import Path



AUSSIE_SLURS = [
    "ranga",
    "bogan",
    "drongo",
    "seppo",
    "eshay",
    "flamin galah",
]


def _scraper(platform: str, query: str):
    if platform == "twitter":
        import snscrape.modules.twitter as sntwitter  # local import to avoid dependency at import time
        return sntwitter.TwitterSearchScraper(query)
    elif platform == "reddit":
        import snscrape.modules.reddit as snreddit  # local import to avoid dependency at import time
        return snreddit.RedditSearchScraper(query)
    else:
        raise ValueError(f"Unknown platform: {platform}")


def scrape_platform(platform: str) -> Path:
    """Scrape posts for all slurs on the given platform."""
    out_dir = Path("data/slang/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    outfile = out_dir / f"{platform}.jsonl"
    logging.info("Saving %s posts to %s", platform, outfile)

    with outfile.open("w", encoding="utf-8") as f:
        for term in AUSSIE_SLURS:
            logging.info("Scraping %s for '%s'", platform, term)
            scraper = _scraper(platform, term)
            count = 0
            try:
                for item in scraper.get_items():
                    data = json.loads(item.json())
                    if platform == "twitter" and "user" in data:
                        user = data.get("user") or {}
                        if isinstance(user, dict):
                            user.pop("username", None)
                            data["user"] = user
                    f.write(json.dumps(data) + "\n")
                    count += 1
                    if count >= 5000:
                        break
            except Exception as e:  # pragma: no cover - network-dependent
                logging.warning("Encountered error: %s", e)
                logging.info("Sleeping due to possible rate limit...")
                time.sleep(60)
            time.sleep(1)
    return outfile


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for platform in ["twitter", "reddit"]:
        scrape_platform(platform)
