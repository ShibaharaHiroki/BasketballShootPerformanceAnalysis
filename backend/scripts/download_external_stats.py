import os
import json
import logging
from core.external_stats import (
    fetch_nba_player_info, 
    fetch_nba_career_stats, 
    fetch_nba_player_awards,
    fetch_nba_shooting_splits,
    fetch_nba_general_splits,
    fetch_player_wiki_summary
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default players to pre-download
DEFAULT_PLAYERS = {
    203999: "Nikola Jokic",
    203507: "Giannis Antetokounmpo",
    203954: "Joel Embiid"
}

CACHE_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "external_stats_cache.json")

def download_stats():
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    
    cache = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            cache = json.load(f)
            
    for pid, pname in DEFAULT_PLAYERS.items():
        str_pid = str(pid)
        logger.info(f"Downloading data for {pname} (ID: {pid})...")
        
        info = fetch_nba_player_info(pid)
        career = fetch_nba_career_stats(pid)
        awards = fetch_nba_player_awards(pid)
        shooting = fetch_nba_shooting_splits(pid)
        general = fetch_nba_general_splits(pid)
        wiki = fetch_player_wiki_summary(pname)
        
        cache[str_pid] = {
            "player_name": pname,
            "player_info": info,
            "career": career,
            "awards": awards,
            "shooting_splits": shooting,
            "general_splits": general,
            "wiki": wiki
        }
        
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
        
    logger.info(f"Successfully downloaded and saved stats for {len(DEFAULT_PLAYERS)} players to {CACHE_FILE}")

if __name__ == "__main__":
    download_stats()
