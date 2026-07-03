"""Quick verification test for react_agent module."""
import sys
sys.path.insert(0, ".")

# Test 1: Import
print("=" * 50)
print("Test 1: Import react_agent module")
try:
    from core.react_agent import (
        run_react_agent,
        parse_action,
        load_stats_cache,
        load_wiki_cache,
    )
    print("  PASS: All imports successful")
except Exception as e:
    print(f"  FAIL: {e}")
    sys.exit(1)

# Test 2: Parse actions
print("\nTest 2: parse_action()")
tests = [
    ("search_wikipedia[Nikola Jokic]", ("search_wikipedia", "Nikola Jokic")),
    ("lookup_wikipedia[MVP]", ("lookup_wikipedia", "MVP")),
    ("search_nba_stats[Giannis Antetokounmpo]", ("search_nba_stats", "Giannis Antetokounmpo")),
    ("get_cluster_data[1]", ("get_cluster_data", "1")),
    ("finish[## Final Answer\nThis is the result]", ("finish", "## Final Answer\nThis is the result")),
]
for input_str, expected in tests:
    result = parse_action(input_str)
    if result == expected:
        print(f"  PASS: {input_str[:40]}...")
    else:
        print(f"  FAIL: {input_str[:40]}...")
        print(f"    Expected: {expected}")
        print(f"    Got:      {result}")

# Test 3: Load stats cache
print("\nTest 3: load_stats_cache()")
try:
    cache = load_stats_cache()
    print(f"  PASS: Loaded {len(cache)} players: {list(cache.keys())}")
except Exception as e:
    print(f"  FAIL: {e}")

# Test 4: Load wiki cache
print("\nTest 4: load_wiki_cache()")
try:
    player_ids = [203999, 203507, 203954]
    player_names = ["Jokic", "Antetokounmpo", "Embiid"]
    wiki = load_wiki_cache(cache, player_ids, player_names)
    for name, text in wiki.items():
        print(f"  {name}: {len(text)} chars")
    print(f"  PASS: {len(wiki)} wiki entries loaded")
except Exception as e:
    print(f"  FAIL: {e}")

# Test 5: Verify routes.py imports
print("\nTest 5: routes.py import check")
try:
    from api.routes import router
    print("  PASS: routes.py imports successfully")
except Exception as e:
    print(f"  FAIL: {e}")

print("\n" + "=" * 50)
print("All tests complete!")
