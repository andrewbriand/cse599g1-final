import json
import requests
import os
import sys

r = requests.post("https://www.codingame.com/services/Leaderboards/getFilteredPuzzleLeaderboard", 
                  json = ["tic-tac-toe", None, "global", {"active": False, "column": "", "filter": ""}])
result = r.json()

player_ids = []
for p in result["users"]:
  player_ids.append(p["agentId"])
player_ids = player_ids[1:20]
print(player_ids)

for player_id in player_ids:
  r = requests.post("https://www.codingame.com/services/gamesPlayersRanking/findLastBattlesByAgentId",
                    json = [player_id, None])
  result = r.json()
  
  gameIds = []
  for p in result:
    if p["done"]:
      gameIds.append(p["gameId"])
  
  for game_id in gameIds:
    r = requests.post(
      'https://www.codingame.com/services/gameResultRemoteService/findByGameId',
      json = [str(game_id), None]
    )
    replay = r.json()
    with open(os.path.join(sys.argv[1], str(game_id)) + '.json', 'w+') as f:
      f.write(json.dumps(replay))
