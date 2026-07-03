"""
ReAct Agent for Basketball Shot Pattern Analysis.

Implements the ReAct (Reasoning + Acting) prompting methodology (Yao et al., 2022)
where the LLM alternates between Thought, Action, and Observation steps to
selectively retrieve information and reason about cluster comparisons.

Tools (Action Space):
  - search_wikipedia[player_name]: First 5 sentences of Wikipedia article
  - lookup_wikipedia[string]: Find next sentence containing string in current page
  - search_nba_stats[player_name]: Career stats summary from NBA API cache
  - get_cluster_data[cluster_id]: Formatted cluster statistics
  - finish[answer]: Return final analysis
"""

import re
import os
import json
import logging
import requests
from typing import Dict, Any, List, Optional, Callable

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────────

MAX_REACT_STEPS = 7
FALLBACK_MAX_STEPS = 10


# ── LLM Client Abstraction ──────────────────────────────────────────────────

class LLMClient:
    """Abstract base for LLM backends."""
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class GeminiClient(LLMClient):
    """Google Gemini API backend."""
    def __init__(self, model_name: str = "gemini-2.5-flash-lite"):
        import google.generativeai as genai
        self._genai = genai
        api_key = os.environ.get("GEMINI_API_KEY", "")
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model_name)

    def generate(self, prompt: str) -> str:
        response = self._model.generate_content(prompt)
        return response.text.strip() if response.text else ""


class OllamaClient(LLMClient):
    """
    Ollama local LLM backend.

    Calls the Ollama REST API (compatible with any machine on the network).
    Default host: http://localhost:11434
    For a Mac on the same network: http://<mac-ip>:11434
    """
    def __init__(
        self,
        model_name: str = "llama3.1:8b",
        host: str = "http://localhost:11434",
        timeout: int = 120,
    ):
        self.model_name = model_name
        self.host = host.rstrip("/")
        self.timeout = timeout
        logger.info(f"OllamaClient: model={model_name}, host={host}")

    def generate(self, prompt: str) -> str:
        url = f"{self.host}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 4096,
            },
        }
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "").strip()
        except requests.exceptions.ConnectionError:
            logger.error(
                f"Cannot connect to Ollama at {self.host}. "
                f"Is Ollama running? Try: OLLAMA_HOST=0.0.0.0 ollama serve"
            )
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.host}. "
                f"Make sure Ollama is running on the target machine."
            )
        except Exception as e:
            logger.error(f"Ollama generate failed: {e}")
            raise


def create_llm_client(
    provider: str = "gemini",
    model_name: Optional[str] = None,
    ollama_host: Optional[str] = None,
) -> LLMClient:
    """
    Factory function to create the appropriate LLM client.

    Args:
        provider: "gemini" or "ollama"
        model_name: Model name (provider-specific)
        ollama_host: Ollama server URL (only for ollama provider)
    """
    if provider == "ollama":
        return OllamaClient(
            model_name=model_name or os.environ.get("OLLAMA_MODEL", "llama3.1:8b"),
            host=ollama_host or os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        )
    else:
        # Default: Gemini
        return GeminiClient(
            model_name=model_name or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite"),
        )

# ── Few-shot ReAct Exemplars ────────────────────────────────────────────────

FEW_SHOT_EXEMPLARS = """Here are example trajectories of how to analyze basketball clusters using the ReAct approach:

--- Example 1 ---
Question: Compare shot pattern Cluster 1 and Cluster 2 for Nikola Jokic and Giannis Antetokounmpo.

Thought 1: I need to first look at the internal cluster statistics to understand the shot pattern differences. Let me start with Cluster 1.
Action 1: get_cluster_data[1]
Observation 1: 【Cluster 1 Stats】Games: 45
By Shot Type:
  - 2PT Field Goal: 312 attempts, FG% 58.3%, EFG% 58.3%
  - 3PT Field Goal: 87 attempts, FG% 34.5%, EFG% 51.7%
By Quarter:
  - 1Q: 105 attempts, FG% 52.4%, EFG% 55.1%
  - 2Q: 98 attempts, FG% 55.1%, EFG% 57.3%
  - 3Q: 102 attempts, FG% 48.0%, EFG% 50.5%
  - 4Q: 94 attempts, FG% 53.2%, EFG% 55.8%

Thought 2: Cluster 1 is paint-heavy with strong 2PT FG% (58.3%). Now let me check Cluster 2.
Action 2: get_cluster_data[2]
Observation 2: 【Cluster 2 Stats】Games: 38
By Shot Type:
  - 2PT Field Goal: 198 attempts, FG% 49.5%, EFG% 49.5%
  - 3PT Field Goal: 156 attempts, FG% 37.2%, EFG% 55.8%
By Quarter:
  - 1Q: 88 attempts, FG% 45.5%, EFG% 52.3%
  - 2Q: 92 attempts, FG% 44.6%, EFG% 53.1%
  - 3Q: 85 attempts, FG% 41.2%, EFG% 48.8%
  - 4Q: 89 attempts, FG% 43.8%, EFG% 51.7%

Thought 3: Cluster 2 has significantly more 3PT attempts (156 vs 87) and lower 2PT FG%. This suggests different offensive strategies. Let me search for Jokic's background to contextualize.
Action 3: search_wikipedia[Nikola Jokic]
Observation 3: Nikola Jokić is a Serbian professional basketball player for the Denver Nuggets of the NBA. A five-time All-Star, he has won three consecutive MVP awards and led the Nuggets to their first championship in 2023. Known for his exceptional passing and court vision as a center, he is considered one of the greatest players in NBA history. Jokić was selected 41st overall in the 2014 NBA draft.

Thought 4: Jokic is a center known for post play and passing. Cluster 1's paint-dominant pattern aligns with his typical style. Let me also check his career stats for shooting context.
Action 4: search_nba_stats[Nikola Jokic]
Observation 4: Career: 9 seasons, 658 GP | PPG 20.7 / RPG 10.6 / APG 7.1 | FG% 55.8% / 3P% 33.7% / FT% 82.1% | Awards: 3x MVP, 5x All-Star, NBA Champion

Thought 5: Jokic's career FG% (55.8%) and relatively low 3P% (33.7%) strongly matches Cluster 1's profile. Cluster 2's perimeter-heavy approach with lower 2PT FG% likely represents games where Antetokounmpo takes more jump shots. Let me get Giannis's stats.
Action 5: search_nba_stats[Giannis Antetokounmpo]
Observation 5: Career: 10 seasons, 732 GP | PPG 23.4 / RPG 9.8 / APG 5.3 | FG% 54.5% / 3P% 28.7% / FT% 70.8% | Awards: 2x MVP, 7x All-Star, NBA Champion, DPOY

Thought 6: Interesting — Giannis has a low 3P% (28.7%), yet Cluster 2 shows more 3PT attempts. This could indicate games where both players were forced into perimeter shooting by defensive schemes. The feature importance data would help clarify which court zones differ most. The data I have is sufficient to write a comprehensive analysis now.
Action 6: finish[## Cluster Comparison Analysis

### Cluster 1: Paint-Dominant Efficiency Pattern
- **Profile**: Heavily interior-focused with 312 2PT attempts vs only 87 3PT attempts
- **Strength**: Elite 2PT FG% of 58.3%, reflecting efficient post-up and drive plays
- **Timing**: Consistent performance across quarters, with slight dip in 3Q (48.0% FG%)
- This pattern aligns closely with Nikola Jokić's career profile — a center who shoots 55.8% from the field over 9 NBA seasons and led Denver to their first championship in 2023. His post play and exceptional court vision make him most dangerous near the rim.

### Cluster 2: Perimeter-Extended Pattern
- **Profile**: More balanced shot distribution with 156 3PT attempts (nearly double Cluster 1)
- **Strength**: Higher EFG% on 3-pointers (55.8%) despite lower overall FG%
- **Timing**: Lower conversion rates across all quarters, suggesting contested or transition shots
- Games in this cluster likely represent possessions where paint access was denied by opposing defenses. Giannis Antetokounmpo, despite his 28.7% career 3P%, has been expanding his perimeter game — these may reflect that developmental trend.

### Key Differences
- **Shot selection**: Cluster 1 favors the paint (78% 2PT), Cluster 2 is more perimeter-oriented (44% 3PT)
- **Efficiency**: Cluster 1 is more efficient on 2PT shots (+8.8 FG%), but Cluster 2 achieves competitive EFG% through threes
- **Quarter consistency**: Cluster 1 maintains steadier performance; Cluster 2 shows more variance

### Tactical Implications
- When facing weak interior defense, lean into the Cluster 1 approach — maximize post touches and drives
- Against teams that pack the paint, the Cluster 2 approach of spacing the floor with threes becomes necessary
- Focus on improving 3Q performance in both clusters, as both show efficiency dips mid-game]

--- Example 2 ---
Question: Compare shot pattern Cluster 1 and Cluster 2 for Joel Embiid.

Thought 1: Let me start by examining the cluster statistics to understand what differentiates the two patterns.
Action 1: get_cluster_data[1]
Observation 1: 【Cluster 1 Stats】Games: 28
By Shot Type:
  - 2PT Field Goal: 245 attempts, FG% 52.7%, EFG% 52.7%
  - 3PT Field Goal: 112 attempts, FG% 35.7%, EFG% 53.6%
By Quarter:
  - 1Q: 92 attempts, FG% 50.0%, EFG% 53.3%
  - 2Q: 88 attempts, FG% 47.7%, EFG% 51.1%
  - 3Q: 95 attempts, FG% 44.2%, EFG% 48.4%
  - 4Q: 82 attempts, FG% 48.8%, EFG% 52.4%

Thought 2: Cluster 1 shows a mixed inside-outside game with decent 3PT volume. Let me check Cluster 2.
Action 2: get_cluster_data[2]
Observation 2: 【Cluster 2 Stats】Games: 22
By Shot Type:
  - 2PT Field Goal: 280 attempts, FG% 56.8%, EFG% 56.8%
  - 3PT Field Goal: 42 attempts, FG% 28.6%, EFG% 42.9%
By Quarter:
  - 1Q: 78 attempts, FG% 57.7%, EFG% 57.7%
  - 2Q: 80 attempts, FG% 55.0%, EFG% 55.0%
  - 3Q: 82 attempts, FG% 51.2%, EFG% 51.2%
  - 4Q: 82 attempts, FG% 54.9%, EFG% 54.9%

Thought 3: Cluster 2 is heavily paint-focused with much fewer 3PT attempts (42 vs 112) and higher 2PT FG% (56.8%). Let me search Wikipedia for Embiid's background.
Action 3: search_wikipedia[Joel Embiid]
Observation 3: Joel Hans Embiid is a Cameroonian-French professional basketball player for the Philadelphia 76ers of the NBA. He is a seven-time All-Star and won the NBA MVP award in 2023. Originally from Yaoundé, Cameroon, Embiid began playing basketball at age 15. He was drafted third overall in 2014 by the 76ers. Known as "The Process," he is considered one of the most dominant centers in the modern NBA.

Thought 4: Embiid is a dominant modern center who can also shoot from outside. The two clusters likely represent his dual playing styles. Let me check his career stats.
Action 4: search_nba_stats[Joel Embiid]
Observation 4: Career: 8 seasons, 433 GP | PPG 27.9 / RPG 11.2 / APG 3.6 | FG% 49.9% / 3P% 33.0% / FT% 85.6% | Awards: MVP, 7x All-Star, 3x All-NBA

Thought 5: Embiid's career 3P% is 33.0%, which is decent for a center. Cluster 1 with 112 3PT attempts at 35.7% matches his shooting capability. Cluster 2 reflects his interior dominance. I have enough context for a complete analysis.
Action 5: finish[## Cluster Comparison: Joel Embiid's Dual Offensive Identity

### Cluster 1: Stretch-Five Mode (28 games)
- **Profile**: Balanced attack with 112 3PT attempts alongside 245 2PT attempts
- **Strength**: Effective 3PT shooting (35.7%) creates floor spacing
- **Trend**: Higher shot volume in 1Q-3Q suggests early establishment of perimeter game
- This pattern showcases Embiid's modern center versatility — drafted 3rd overall in 2014, the Cameroonian-French seven-time All-Star has developed into a legitimate perimeter threat, shooting 33.0% from three across his career.

### Cluster 2: Post-Dominant Mode (22 games)
- **Profile**: Interior-focused with only 42 3PT attempts vs 280 2PT attempts
- **Strength**: Exceptional 2PT FG% (56.8%) with consistent quarterly performance
- **Trend**: Steady efficiency across all four quarters reflects controlled, methodical post play
- These represent games where "The Process" operates primarily in the paint, leveraging his 7'0" frame and footwork. His 2023 MVP campaign was built on this dominant interior presence.

### Key Differences
- **Three-point volume**: Cluster 1 attempts nearly 3x more threes (112 vs 42)
- **Interior efficiency**: Cluster 2 is +4.1% more efficient on 2PT shots
- **Overall approach**: Cluster 1 = face-up/perimeter; Cluster 2 = back-to-basket

### Tactical Implications
- Monitor game matchups: against smaller lineups, the Cluster 2 approach exploits size advantage
- Against drop-coverage bigs, the Cluster 1 perimeter game punishes conservative defense
- Both modes maintain solid 4Q performance, indicating clutch reliability regardless of style]

"""


# ── Tool Implementations ───────────────────────────────────────────────────

def _build_wiki_search_tool(
    player_names: List[str],
    wiki_cache: Dict[str, str],
) -> Callable:
    """Build the search_wikipedia tool backed by cache."""

    # Track current page for lookup
    state = {"current_page": "", "current_player": ""}

    def search_wikipedia(entity: str) -> str:
        # Find best matching player
        entity_lower = entity.lower().strip()
        matched_name = None
        for name in player_names:
            if entity_lower in name.lower() or name.lower() in entity_lower:
                matched_name = name
                break

        if not matched_name:
            # Try partial matching
            for name in player_names:
                name_parts = name.lower().split()
                if any(part in entity_lower for part in name_parts):
                    matched_name = name
                    break

        if not matched_name:
            return f"Could not find Wikipedia article for '{entity}'. Available players: {', '.join(player_names)}"

        full_text = wiki_cache.get(matched_name, "")
        if not full_text:
            return f"No Wikipedia data available for {matched_name}."

        state["current_page"] = full_text
        state["current_player"] = matched_name

        # Return first 5 sentences
        sentences = re.split(r'(?<=[.!?])\s+', full_text.strip())
        intro = " ".join(sentences[:5])
        return intro if intro else f"No Wikipedia data available for {matched_name}."

    def lookup_wikipedia(search_string: str) -> str:
        if not state["current_page"]:
            return "No Wikipedia page is currently loaded. Use search_wikipedia first."

        search_lower = search_string.lower()
        sentences = re.split(r'(?<=[.!?])\s+', state["current_page"].strip())
        for sent in sentences:
            if search_lower in sent.lower():
                return sent
        return f"The string '{search_string}' was not found in the current Wikipedia page for {state['current_player']}."

    return search_wikipedia, lookup_wikipedia


def _build_nba_stats_tool(
    player_names: List[str],
    stats_cache: Dict[str, Dict],
) -> Callable:
    """Build the search_nba_stats tool backed by cache."""

    def search_nba_stats(player_name: str) -> str:
        name_lower = player_name.lower().strip()

        # Find matching player in cache
        matched_pid = None
        for pid, data in stats_cache.items():
            cached_name = data.get("player_name", "")
            if name_lower in cached_name.lower() or cached_name.lower() in name_lower:
                matched_pid = pid
                break

        if not matched_pid:
            # Partial match
            for pid, data in stats_cache.items():
                cached_name = data.get("player_name", "")
                name_parts = name_lower.split()
                if any(part in cached_name.lower() for part in name_parts):
                    matched_pid = pid
                    break

        if not matched_pid:
            return f"No NBA stats found for '{player_name}'. Available: {', '.join(player_names)}"

        data = stats_cache[matched_pid]
        lines = []

        # Basic info
        info = data.get("player_info")
        if info:
            lines.append(
                f"Position: {info.get('position', 'N/A')} | "
                f"Height: {info.get('height', 'N/A')} | "
                f"Weight: {info.get('weight', 'N/A')}lbs | "
                f"Team: {info.get('team_city', '')} {info.get('team', '')} "
                f"(#{info.get('jersey', '')})"
            )
            lines.append(
                f"Draft: {info.get('draft_year', 'N/A')} "
                f"Round {info.get('draft_round', 'N/A')} "
                f"Pick {info.get('draft_number', 'N/A')} | "
                f"Country: {info.get('country', 'N/A')}"
            )

        # Career stats
        career = data.get("career")
        if career:
            lines.append(
                f"Career: {career.get('seasons_played', 'N/A')} seasons, "
                f"{career.get('career_games', 0)} GP | "
                f"PPG {career.get('career_ppg', 0)} / "
                f"RPG {career.get('career_rpg', 0)} / "
                f"APG {career.get('career_apg', 0)} | "
                f"FG% {career.get('career_fg_pct', 0)}% / "
                f"3P% {career.get('career_fg3_pct', 0)}% / "
                f"FT% {career.get('career_ft_pct', 0)}%"
            )
            lines.append(
                f"Latest ({career.get('latest_season', 'N/A')} / "
                f"{career.get('latest_team', 'N/A')}): "
                f"PPG {career.get('latest_ppg', 0)} / "
                f"RPG {career.get('latest_rpg', 0)} / "
                f"APG {career.get('latest_apg', 0)}, "
                f"FG% {career.get('latest_fg_pct', 0)}% / "
                f"3P% {career.get('latest_fg3_pct', 0)}%"
            )

        # Awards (compact)
        awards = data.get("awards", [])
        if awards:
            # Deduplicate and summarize
            from collections import Counter
            award_counts = Counter()
            for award_str in awards:
                # Extract base award name (before the " - season" part)
                base = award_str.split(" - ")[0].strip()
                award_counts[base] += 1
            compact_awards = []
            for award, count in award_counts.most_common(10):
                if count > 1:
                    compact_awards.append(f"{count}x {award}")
                else:
                    compact_awards.append(award)
            lines.append(f"Awards: {', '.join(compact_awards)}")

        return "\n".join(lines) if lines else f"No detailed stats available for {player_name}."

    return search_nba_stats


def _build_cluster_data_tool(
    cluster_texts: Dict[str, str],
    importance_text: str,
    composition_text: str,
) -> Callable:
    """Build the get_cluster_data tool."""

    def get_cluster_data(cluster_id: str) -> str:
        cid = cluster_id.strip()
        if cid == "1":
            result = cluster_texts.get("1", "No data for Cluster 1.")
            if importance_text:
                result += "\n" + importance_text
            if composition_text:
                result += "\n" + composition_text
            return result
        elif cid == "2":
            return cluster_texts.get("2", "No data for Cluster 2.")
        else:
            return f"Invalid cluster_id '{cid}'. Use 1 or 2."

    return get_cluster_data


# ── Action Parser ───────────────────────────────────────────────────────────

def parse_action(text: str) -> Optional[tuple]:
    """
    Parse an action string like 'search_wikipedia[Nikola Jokic]' or 'finish[answer text]'.
    Returns (action_name, argument) or None.
    """
    # Match action_name[argument] — argument can be multi-line for finish
    match = re.search(
        r'(search_wikipedia|lookup_wikipedia|search_nba_stats|get_cluster_data|finish)\[(.+)',
        text,
        re.DOTALL
    )
    if not match:
        return None

    action_name = match.group(1)
    rest = match.group(2)

    # Handle the closing bracket — for finish, the argument can span multiple lines
    if action_name == "finish":
        # Find the last ']' to handle multi-line content
        bracket_depth = 1
        for i, char in enumerate(rest):
            if char == '[':
                bracket_depth += 1
            elif char == ']':
                bracket_depth -= 1
                if bracket_depth == 0:
                    return (action_name, rest[:i])
        # If no matching bracket, take everything
        return (action_name, rest.rstrip(']').strip())
    else:
        # Simple single-line tools
        bracket_idx = rest.find(']')
        if bracket_idx != -1:
            return (action_name, rest[:bracket_idx].strip())
        return (action_name, rest.strip().rstrip(']'))


# ── Main ReAct Loop ─────────────────────────────────────────────────────────

def run_react_agent(
    cluster1_text: str,
    cluster2_text: str,
    importance_text: str,
    composition_text: str,
    player_names: List[str],
    league_label: str,
    stats_cache: Dict[str, Dict],
    wiki_cache: Dict[str, str],
    model_name: str = "gemini-2.5-flash-lite",
    max_steps: int = MAX_REACT_STEPS,
    llm_provider: str = "gemini",
    ollama_host: Optional[str] = None,
) -> str:
    """
    Run the ReAct agent loop for cluster comparison analysis.

    Args:
        llm_provider: "gemini" or "ollama"
        ollama_host: Ollama server URL (e.g. "http://192.168.1.10:11434")

    Returns the final analysis text.
    """
    # Build tools
    search_wikipedia, lookup_wikipedia = _build_wiki_search_tool(
        player_names, wiki_cache
    )
    search_nba_stats = _build_nba_stats_tool(player_names, stats_cache)
    get_cluster_data = _build_cluster_data_tool(
        {"1": cluster1_text, "2": cluster2_text},
        importance_text,
        composition_text,
    )

    tool_map = {
        "search_wikipedia": search_wikipedia,
        "lookup_wikipedia": lookup_wikipedia,
        "search_nba_stats": search_nba_stats,
        "get_cluster_data": get_cluster_data,
    }

    # System prompt
    players_str = ", ".join(player_names) if player_names else "(unspecified)"

    system_prompt = f"""You are an expert basketball shot pattern analyst for the {league_label}.
You analyze shot pattern clusters identified by the TULCA dimensionality reduction method.
NOTE: The cluster tracking data is specifically from the 2022-2023 season.

You have access to the following tools:
- search_wikipedia[player_name]: Returns the first 5 sentences from the player's Wikipedia page.
- lookup_wikipedia[string]: Returns the next sentence in the current Wikipedia page containing the given string. You must call search_wikipedia first.
- search_nba_stats[player_name]: Returns career statistics, awards, and shooting data from NBA official records.
- get_cluster_data[cluster_id]: Returns the formatted statistics for cluster 1 or 2 (shot types, timing, feature importance).
- finish[answer]: Terminates the analysis and returns your final answer. The answer should be the complete analysis.

You MUST follow this format strictly:

Thought 1: [your reasoning about what to do next]
Action 1: [tool_name[argument]]

After receiving an Observation, continue with:

Thought 2: [your reasoning based on the observation]
Action 2: [tool_name[argument]]

...and so on until you have enough information, then use:

Thought N: [final synthesis reasoning]
Action N: finish[your complete analysis in markdown format]

IMPORTANT RULES:
- Always start by examining the cluster data using get_cluster_data
- Use search_wikipedia and search_nba_stats to enrich the analysis with player context
- Naturally weave external knowledge into the cluster analysis — do NOT use explicit headings like "External Statistics"
- Your final answer inside finish[] MUST be in English
- Use bullet points and headings for readability
- Focus on: (1) each cluster's shot pattern characteristics, (2) main differences, (3) key court area differences from feature importance, (4) actionable tactical implications
- Do NOT explicitly mention player composition breakdown in the output (use it only as background context)
"""

    # Build initial prompt
    prompt_parts = [system_prompt, "\n", FEW_SHOT_EXEMPLARS, "\n"]
    prompt_parts.append(
        f"Now analyze the following:\n"
        f"Question: Compare shot pattern Cluster 1 and Cluster 2 for {players_str}.\n\n"
    )

    # Create the appropriate LLM client
    llm = create_llm_client(
        provider=llm_provider,
        model_name=model_name,
        ollama_host=ollama_host,
    )

    trajectory = "".join(prompt_parts)
    final_answer = None

    for step in range(1, max_steps + 1):
        try:
            # Generate the next Thought + Action
            generated = llm.generate(trajectory)
            if not generated:
                logger.warning(f"ReAct step {step}: empty response")
                break

            logger.info(f"ReAct step {step} generated:\n{generated[:200]}...")

            # Parse the action from generated text
            action = parse_action(generated)

            if not action:
                # Model didn't produce a valid action — append text and retry
                trajectory += generated + "\n"
                trajectory += f"\nPlease provide your next step in the correct format: Thought {step}: ... Action {step}: tool_name[argument]\n"
                continue

            action_name, argument = action

            # Handle finish action
            if action_name == "finish":
                final_answer = argument
                break

            # Execute the tool
            if action_name in tool_map:
                observation = tool_map[action_name](argument)
            else:
                observation = f"Unknown tool: {action_name}"

            # Extract just the Thought and Action from the generated text
            # (truncate anything after the action for cleanliness)
            action_pattern = re.search(
                rf'{re.escape(action_name)}\[.*?\]',
                generated,
                re.DOTALL,
            )
            if action_pattern:
                clean_generated = generated[:action_pattern.end()]
            else:
                clean_generated = generated

            # Append to trajectory
            trajectory += clean_generated + "\n"
            trajectory += f"Observation {step}: {observation}\n\n"

        except Exception as e:
            logger.error(f"ReAct step {step} failed: {e}")
            break

    # If we didn't get a finish action, fall back to direct prompt
    if final_answer is None:
        logger.warning("ReAct did not produce finish action, falling back to direct synthesis")
        final_answer = _fallback_synthesis(
            trajectory, llm, players_str, league_label
        )

    return final_answer


def _fallback_synthesis(
    trajectory: str,
    llm: LLMClient,
    players_str: str,
    league_label: str,
) -> str:
    """
    Fallback: if the ReAct loop didn't converge with finish[],
    ask the model to synthesize from the trajectory so far.
    (Corresponds to ReAct → CoT-SC backoff from the paper §3.2)
    """
    fallback_prompt = (
        f"{trajectory}\n\n"
        f"Based on all the observations above, provide a complete analysis "
        f"comparing the two clusters for {players_str} ({league_label}). "
        f"Output in English using markdown with bullet points and headings. "
        f"Focus on: (1) each cluster's shot pattern characteristics, "
        f"(2) main differences, (3) key court area differences, "
        f"(4) actionable tactical implications. "
        f"Naturally integrate player background into the analysis."
    )
    try:
        result = llm.generate(fallback_prompt)
        return result if result else "Analysis could not be generated."
    except Exception as e:
        logger.error(f"Fallback synthesis failed: {e}")
        return "Analysis could not be generated due to an error."


# ── Cache Loaders (for use by routes.py) ────────────────────────────────────

def load_stats_cache() -> Dict[str, Dict]:
    """Load the pre-downloaded external_stats_cache.json."""
    cache_file = os.path.join(
        os.path.dirname(__file__), "..", "data", "external_stats_cache.json"
    )
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}


def load_wiki_cache(
    stats_cache: Dict[str, Dict],
    player_ids: List[int],
    player_names: List[str],
) -> Dict[str, str]:
    """
    Extract Wikipedia text from the stats cache, keyed by player name.
    """
    wiki_cache = {}
    for pid, pname in zip(player_ids, player_names):
        str_pid = str(pid)
        if str_pid in stats_cache:
            wiki_text = stats_cache[str_pid].get("wiki", "")
            if wiki_text:
                wiki_cache[pname] = wiki_text
    return wiki_cache
