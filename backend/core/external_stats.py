"""
External statistics retrieval module.

Fetches comprehensive player data from nba_api and biographical info
from Wikipedia, inspired by Sportify's ReAct tool suite (§4.3).
"""

import logging
from functools import lru_cache
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ── NBA Comprehensive Player Stats via nba_api ─────────────────────────────

@lru_cache(maxsize=64)
def fetch_nba_player_info(player_id: int) -> Optional[Dict]:
    """
    Fetch basic player info: height, weight, position, draft, birthdate, etc.
    Uses CommonPlayerInfo endpoint.
    """
    try:
        from nba_api.stats.endpoints import CommonPlayerInfo
        import time
        time.sleep(0.5)

        info = CommonPlayerInfo(player_id=str(player_id), timeout=10)
        dfs = info.get_data_frames()
        if not dfs or dfs[0].empty:
            return None

        row = dfs[0].iloc[0]
        result = {
            "display_name": str(row.get("DISPLAY_FIRST_LAST", "")),
            "birthdate": str(row.get("BIRTHDATE", ""))[:10],  # YYYY-MM-DD
            "height": str(row.get("HEIGHT", "")),
            "weight": str(row.get("WEIGHT", "")),
            "position": str(row.get("POSITION", "")),
            "jersey": str(row.get("JERSEY", "")),
            "team": str(row.get("TEAM_NAME", "")),
            "team_city": str(row.get("TEAM_CITY", "")),
            "team_abbreviation": str(row.get("TEAM_ABBREVIATION", "")),
            "country": str(row.get("COUNTRY", "")),
            "school": str(row.get("SCHOOL", "")),
            "draft_year": str(row.get("DRAFT_YEAR", "")),
            "draft_round": str(row.get("DRAFT_ROUND", "")),
            "draft_number": str(row.get("DRAFT_NUMBER", "")),
            "from_year": str(row.get("FROM_YEAR", "")),
            "to_year": str(row.get("TO_YEAR", "")),
            "seasons_exp": str(row.get("SEASON_EXP", "")),
        }

        # Headline stats (if available in second DataFrame)
        if len(dfs) > 1 and not dfs[1].empty:
            headline = dfs[1].iloc[0]
            result["headline_pts"] = str(headline.get("PTS", ""))
            result["headline_reb"] = str(headline.get("REB", ""))
            result["headline_ast"] = str(headline.get("AST", ""))
            result["headline_pie"] = str(headline.get("PIE", ""))

        return result
    except Exception as e:
        logger.warning(f"Failed to fetch CommonPlayerInfo for {player_id}: {e}")
        return None


@lru_cache(maxsize=64)
def fetch_nba_career_stats(player_id: int) -> Optional[Dict]:
    """
    Fetch career and per-season stats via PlayerCareerStats endpoint.
    Returns career totals, per-game averages, and the latest season line.
    """
    try:
        from nba_api.stats.endpoints import PlayerCareerStats
        import time
        time.sleep(0.5)

        career = PlayerCareerStats(player_id=str(player_id), timeout=10)
        dfs = career.get_data_frames()
        if not dfs or dfs[0].empty:
            return None

        df_season = dfs[0]  # SeasonTotalsRegularSeason
        latest = df_season.iloc[-1]
        total_gp = df_season["GP"].sum()
        if total_gp == 0:
            return None

        result = {
            "seasons_played": len(df_season),
            "latest_season": str(latest.get("SEASON_ID", "N/A")),
            "latest_team": str(latest.get("TEAM_ABBREVIATION", "N/A")),
            "career_games": int(total_gp),
            "career_ppg": round(df_season["PTS"].sum() / total_gp, 1),
            "career_rpg": round(df_season["REB"].sum() / total_gp, 1),
            "career_apg": round(df_season["AST"].sum() / total_gp, 1),
            "career_spg": round(df_season["STL"].sum() / total_gp, 1),
            "career_bpg": round(df_season["BLK"].sum() / total_gp, 1),
            "career_topg": round(df_season["TOV"].sum() / total_gp, 1),
            "career_fg_pct": round(df_season["FGM"].sum() / max(df_season["FGA"].sum(), 1) * 100, 1),
            "career_fg3_pct": round(df_season["FG3M"].sum() / max(df_season["FG3A"].sum(), 1) * 100, 1),
            "career_ft_pct": round(df_season["FTM"].sum() / max(df_season["FTA"].sum(), 1) * 100, 1),
            # Latest season per-game
            "latest_gp": int(latest.get("GP", 0)),
            "latest_ppg": round(int(latest.get("PTS", 0)) / max(int(latest.get("GP", 1)), 1), 1),
            "latest_rpg": round(int(latest.get("REB", 0)) / max(int(latest.get("GP", 1)), 1), 1),
            "latest_apg": round(int(latest.get("AST", 0)) / max(int(latest.get("GP", 1)), 1), 1),
            "latest_fg_pct": round(float(latest.get("FG_PCT", 0)) * 100, 1),
            "latest_fg3_pct": round(float(latest.get("FG3_PCT", 0)) * 100, 1),
            "latest_ft_pct": round(float(latest.get("FT_PCT", 0)) * 100, 1),
        }

        # Per-season history for trend analysis
        season_history = []
        for _, row in df_season.iterrows():
            gp = int(row.get("GP", 0))
            if gp == 0:
                continue
            season_history.append({
                "season": str(row.get("SEASON_ID", "")),
                "team": str(row.get("TEAM_ABBREVIATION", "")),
                "gp": gp,
                "ppg": round(int(row.get("PTS", 0)) / gp, 1),
                "rpg": round(int(row.get("REB", 0)) / gp, 1),
                "apg": round(int(row.get("AST", 0)) / gp, 1),
                "fg_pct": round(float(row.get("FG_PCT", 0)) * 100, 1),
                "fg3_pct": round(float(row.get("FG3_PCT", 0)) * 100, 1),
            })
        result["season_history"] = season_history

        return result
    except Exception as e:
        logger.warning(f"Failed to fetch PlayerCareerStats for {player_id}: {e}")
        return None


@lru_cache(maxsize=64)
def fetch_nba_player_awards(player_id: int) -> Optional[List[str]]:
    """
    Fetch player awards (MVP, All-Star, All-NBA, etc.) via PlayerAwards endpoint.
    Returns a list of award description strings.
    """
    try:
        from nba_api.stats.endpoints import PlayerAwards
        import time
        time.sleep(0.5)

        awards = PlayerAwards(player_id=str(player_id), timeout=10)
        dfs = awards.get_data_frames()
        if not dfs or dfs[0].empty:
            return None

        df = dfs[0]
        award_list = []
        for _, row in df.iterrows():
            desc = str(row.get("DESCRIPTION", ""))
            team = str(row.get("TEAM", ""))
            season = str(row.get("SEASON", ""))
            subtype = str(row.get("SUBTYPE1", ""))
            entry = desc
            if subtype:
                entry += f" ({subtype})"
            if season:
                entry += f" - {season}"
            award_list.append(entry)

        return award_list if award_list else None
    except Exception as e:
        logger.warning(f"Failed to fetch PlayerAwards for {player_id}: {e}")
        return None


@lru_cache(maxsize=64)
def fetch_nba_shooting_splits(player_id: int) -> Optional[Dict]:
    """
    Fetch shooting splits (by distance, area, zone) via PlayerDashboardByShootingSplits.
    """
    try:
        from nba_api.stats.endpoints import PlayerDashboardByShootingSplits
        import time
        time.sleep(0.5)

        splits = PlayerDashboardByShootingSplits(
            player_id=str(player_id), timeout=10
        )
        dfs = splits.get_data_frames()
        result = {}

        # Shot area splits (index 2 typically)
        for i, df in enumerate(dfs):
            if df.empty:
                continue
            # Identify the data by column inspection
            if "GROUP_VALUE" in df.columns:
                group_name = str(df.attrs.get("name", f"split_{i}"))
                entries = []
                for _, row in df.iterrows():
                    gv = str(row.get("GROUP_VALUE", ""))
                    fga = int(row.get("FGA", 0))
                    fg_pct = round(float(row.get("FG_PCT", 0)) * 100, 1)
                    fg3a = int(row.get("FG3A", 0))
                    fg3_pct = round(float(row.get("FG3_PCT", 0)) * 100, 1) if fg3a > 0 else 0
                    entries.append({
                        "group": gv,
                        "fga": fga,
                        "fg_pct": fg_pct,
                        "fg3a": fg3a,
                        "fg3_pct": fg3_pct,
                    })
                if entries:
                    result[f"split_{i}"] = entries

        return result if result else None
    except Exception as e:
        logger.warning(f"Failed to fetch ShootingSplits for {player_id}: {e}")
        return None





@lru_cache(maxsize=64)
def fetch_nba_general_splits(player_id: int) -> Optional[Dict]:
    """
    Fetch general dashboard splits (overall, home/away, wins/losses, etc.)
    via PlayerDashboardByGeneralSplits.
    """
    try:
        from nba_api.stats.endpoints import PlayerDashboardByGeneralSplits
        import time
        time.sleep(0.5)

        dash = PlayerDashboardByGeneralSplits(
            player_id=str(player_id), timeout=10
        )
        dfs = dash.get_data_frames()
        result = {}

        # First dataframe is OverallPlayerDashboard
        if dfs and not dfs[0].empty:
            row = dfs[0].iloc[0]
            result["overall"] = {
                "gp": int(row.get("GP", 0)),
                "min": round(float(row.get("MIN", 0)), 1),
                "pts": round(float(row.get("PTS", 0)), 1),
                "reb": round(float(row.get("REB", 0)), 1),
                "ast": round(float(row.get("AST", 0)), 1),
                "stl": round(float(row.get("STL", 0)), 1),
                "blk": round(float(row.get("BLK", 0)), 1),
                "tov": round(float(row.get("TOV", 0)), 1),
                "plus_minus": round(float(row.get("PLUS_MINUS", 0)), 1),
            }

        # Location splits (Home/Away) - typically index 1
        if len(dfs) > 1 and not dfs[1].empty:
            location_splits = []
            for _, row in dfs[1].iterrows():
                location_splits.append({
                    "group": str(row.get("GROUP_VALUE", "")),
                    "gp": int(row.get("GP", 0)),
                    "pts": round(float(row.get("PTS", 0)), 1),
                    "fg_pct": round(float(row.get("FG_PCT", 0)) * 100, 1),
                })
            result["by_location"] = location_splits

        # Win/Loss splits - typically index 3
        if len(dfs) > 3 and not dfs[3].empty:
            wl_splits = []
            for _, row in dfs[3].iterrows():
                wl_splits.append({
                    "group": str(row.get("GROUP_VALUE", "")),
                    "gp": int(row.get("GP", 0)),
                    "pts": round(float(row.get("PTS", 0)), 1),
                    "fg_pct": round(float(row.get("FG_PCT", 0)) * 100, 1),
                })
            result["by_win_loss"] = wl_splits

        return result if result else None
    except Exception as e:
        logger.warning(f"Failed to fetch GeneralSplits for {player_id}: {e}")
        return None


# ── Wikipedia full article content ──────────────────────────────────────────

@lru_cache(maxsize=64)
def fetch_player_wiki_summary(player_name: str) -> str:
    """
    Fetch the full Wikipedia article content for the given player name.
    Returns the complete page text, or empty string on failure.
    Uses direct requests to avoid wikipedia package JSONDecodeError blocks.
    """
    try:
        import requests
        import urllib.parse
        import unicodedata

        headers = {
            "User-Agent": "BasketballTrackingApp/1.0 (contact@example.com)"
        }

        # Step 1: Search for the title
        search_query = urllib.parse.quote(f"{player_name} basketball")
        search_url = f"https://en.wikipedia.org/w/api.php?action=query&format=json&list=search&srsearch={search_query}&utf8=1&srlimit=3"
        
        search_res = requests.get(search_url, headers=headers, timeout=10)
        search_res.raise_for_status()
        search_data = search_res.json()
        
        search_results = search_data.get("query", {}).get("search", [])
        if not search_results:
            return ""

        # Find best matching title
        best_title = None
        target_name_parts = [p.lower() for p in player_name.split()]
        
        for item in search_results:
            title = item["title"]
            # Normalize accents (e.g. Jokić -> Jokic)
            norm_title = unicodedata.normalize('NFKD', title).encode('ascii', 'ignore').decode('ascii').lower()
            
            # If any part of the player name matches the title, consider it a match
            if any(part in norm_title for part in target_name_parts):
                best_title = title
                break
                
        if not best_title:
            best_title = search_results[0]["title"]  # Fallback to first

        # Step 2: Fetch extract
        title_encoded = urllib.parse.quote(best_title)
        query_url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=true&titles={title_encoded}&format=json"
        
        page_res = requests.get(query_url, headers=headers, timeout=10)
        page_res.raise_for_status()
        page_data = page_res.json()
        
        pages = page_data.get("query", {}).get("pages", {})
        for page_id, page_info in pages.items():
            if "extract" in page_info:
                return page_info["extract"]

        return ""

    except Exception as e:
        logger.warning(f"Failed to fetch Wikipedia content for '{player_name}': {e}")
        return ""


# ── Orchestrator ────────────────────────────────────────────────────────────

def format_player_stats(
    player_name: str,
    player_info: Optional[Dict],
    career: Optional[Dict],
    awards: Optional[List[str]],
    shooting_splits: Optional[Dict],
    general_splits: Optional[Dict],
    wiki: str,
) -> str:
    """Format a single player's comprehensive external data as a text block."""
    lines = [f"■ {player_name}"]

    # ─ Basic Info ─
    if player_info:
        lines.append("  [Basic Information]")
        lines.append(
            f"  Position: {player_info['position']} | "
            f"Height: {player_info['height']} | Weight: {player_info['weight']}lbs"
        )
        lines.append(
            f"  Team: {player_info['team_city']} {player_info['team']} "
            f"(#{player_info['jersey']})"
        )
        lines.append(
            f"  Hometown/Country: {player_info['country']} | School: {player_info['school']}"
        )
        lines.append(
            f"  Draft: {player_info['draft_year']} Round "
            f"{player_info['draft_round']} Pick {player_info['draft_number']}"
        )
        lines.append(
            f"  Birthdate: {player_info['birthdate']} | "
            f"NBA Experience: {player_info['seasons_exp']} years "
            f"({player_info['from_year']}–{player_info['to_year']})"
        )
        # Headline stats
        if "headline_pts" in player_info:
            lines.append(
                f"  Headline: PTS {player_info['headline_pts']} / "
                f"REB {player_info['headline_reb']} / "
                f"AST {player_info['headline_ast']} / "
                f"PIE {player_info['headline_pie']}"
            )

    # ─ Career Stats ─
    if career:
        lines.append("  [Career Stats]")
        lines.append(
            f"  Total: {career['seasons_played']} seasons, "
            f"{career['career_games']} games played"
        )
        lines.append(
            f"  Career Averages: PPG {career['career_ppg']} / "
            f"RPG {career['career_rpg']} / APG {career['career_apg']} / "
            f"SPG {career['career_spg']} / BPG {career['career_bpg']} / "
            f"TOPG {career['career_topg']}"
        )
        lines.append(
            f"  Career Shooting: FG% {career['career_fg_pct']}% / "
            f"3P% {career['career_fg3_pct']}% / FT% {career['career_ft_pct']}%"
        )
        lines.append(
            f"  Latest Season ({career['latest_season']} / {career['latest_team']}): "
            f"{career['latest_gp']} games, "
            f"PPG {career['latest_ppg']} / RPG {career['latest_rpg']} / "
            f"APG {career['latest_apg']}, "
            f"FG% {career['latest_fg_pct']}% / 3P% {career['latest_fg3_pct']}% / "
            f"FT% {career['latest_ft_pct']}%"
        )
        # Season history
        if career.get("season_history"):
            lines.append("  Season History:")
            for s in career["season_history"]:
                lines.append(
                    f"    {s['season']} ({s['team']}): {s['gp']} GP, "
                    f"PPG {s['ppg']}, RPG {s['rpg']}, APG {s['apg']}, "
                    f"FG% {s['fg_pct']}%, 3P% {s['fg3_pct']}%"
                )

    # ─ Awards ─
    if awards:
        lines.append("  [Awards]")
        for award in awards:
            lines.append(f"    - {award}")



    # ─ Shooting Splits ─
    if shooting_splits:
        lines.append("  [Shooting Splits]")
        for split_name, entries in shooting_splits.items():
            for e in entries:
                lines.append(
                    f"    {e['group']}: FGA {e['fga']}, "
                    f"FG% {e['fg_pct']}%"
                    + (f", 3PA {e['fg3a']}, 3P% {e['fg3_pct']}%" if e['fg3a'] > 0 else "")
                )

    # ─ General Splits ─
    if general_splits:
        if "by_location" in general_splits:
            lines.append("  [Home/Away]")
            for s in general_splits["by_location"]:
                lines.append(
                    f"    {s['group']}: {s['gp']} GP, "
                    f"PTS {s['pts']}, FG% {s['fg_pct']}%"
                )
        if "by_win_loss" in general_splits:
            lines.append("  [Wins/Losses]")
            for s in general_splits["by_win_loss"]:
                lines.append(
                    f"    {s['group']}: {s['gp']} GP, "
                    f"PTS {s['pts']}, FG% {s['fg_pct']}%"
                )

    # ─ Wikipedia ─
    if wiki:
        lines.append(f"  [Wikipedia (Full Article)]\n  {wiki}")

    if not any([player_info, career, awards, shooting_splits, general_splits, wiki]):
        lines.append("  (No external data available)")

    return "\n".join(lines)


def fetch_all_player_stats(
    player_ids: List[int],
    player_names: List[str],
    league: str = "nba",
) -> str:
    """
    Fetch comprehensive external stats for all players and return a
    formatted text block ready for injection into the LLM prompt.

    Now reads from pre-downloaded data/external_stats_cache.json.
    For B.League players, external retrieval is skipped.
    """
    if league == "bleague":
        return ""

    if not player_ids or not player_names:
        return ""

    import os
    import json
    
    # Load cache
    cache_file = os.path.join(os.path.dirname(__file__), "..", "data", "external_stats_cache.json")
    cache = {}
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            try:
                cache = json.load(f)
            except json.JSONDecodeError:
                pass

    blocks = []
    seen_ids = set()

    for pid, pname in zip(player_ids, player_names):
        if pid in seen_ids:
            continue
        seen_ids.add(pid)

        if pname.startswith("Player_"):
            continue

        str_pid = str(pid)
        if str_pid in cache:
            data = cache[str_pid]
            blocks.append(format_player_stats(
                pname, 
                data.get("player_info"), 
                data.get("career"), 
                data.get("awards"),
                data.get("shooting_splits"), 
                data.get("general_splits"), 
                data.get("wiki")
            ))
        else:
            # If not in cache, fallback to empty stats for this player
            blocks.append(format_player_stats(pname, None, None, None, None, None, ""))

    if not blocks:
        return ""

    header = "\n[External Statistics Data (Pre-downloaded data)]"
    return header + "\n" + "\n\n".join(blocks)
