from src.games.leduc import LeducHoldem

game = LeducHoldem()
info_sets = set()
def traverse(hist):
    if game.is_terminal(hist):
        return
    player = game.current_player(hist)
    info = game.info_set_key(hist, player)
    info_sets.add(info)
    for a in game.legal_actions(hist):
        traverse(game.apply_action(hist, a))

for h, _ in game.initial_histories():
    traverse(h)
print(f"Info sets total: {len(info_sets)}")