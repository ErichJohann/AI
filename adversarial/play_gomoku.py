import time
import argparse
from colorama import Fore, Style, init
init(autoreset=True)

from board_gomoku import Gomoku, COLS, ROWS

# Importações específicas
from mcts import mcts
from minimax import minimax_with_hef


# ========================
# Função heurística Minimax
# ========================
def evaluate_gomoku(board, player):
    opponent = 'O' if player == 'X' else 'X'
    score = 0

    def evaluate_window(window):
        if window.count(player) == 5:
            return 1000
        elif window.count(player) == 4 and window.count(' ') == 1:
            return 100
        elif window.count(player) == 3 and window.count(' ') == 2:
            return 25
        elif window.count(opponent) == 4 and window.count(' ') == 1:
            return -170
        elif window.count(opponent) == 3 and window.count(' ') == 2:
            return -30
        return 0

    rows = len(board)
    cols = len(board[0])

    # horizontais
    for r in range(rows):
        for c in range(cols - 4):
            score += evaluate_window(board[r][c:c+5])
    # verticais
    for r in range(rows - 4):
        for c in range(cols):
            window = [board[r+i][c] for i in range(5)]
            score += evaluate_window(window)
    # diagonais \
    for r in range(rows - 4):
        for c in range(cols - 4):
            window = [board[r+i][c+i] for i in range(5)]
            score += evaluate_window(window)
    # diagonais /
    for r in range(4, rows):
        for c in range(cols - 4):
            window = [board[r-i][c+i] for i in range(5)]
            score += evaluate_window(window)

    return score


def best_move_minimax(game, depth=1):
    player = game.current
    best_score = float('-inf')
    move_choice = None
    for move in game.available_moves():
        new_game = game.copy()
        new_game.make_move(move)
        score = minimax_with_hef(
            game=new_game,
            depth=depth - 1,
            maximizing=False,
            player=player,
            evaluate_fn=evaluate_gomoku
        )
        if score > best_score:
            best_score = score
            move_choice = move
    return move_choice


# ========================
# Função principal
# ========================
def play(algorithm: str):
    game = Gomoku()
    human = input("Escolha seu lado (X ou O): ").strip().upper()
    assert human in ['X', 'O']
    ai = 'O' if human == 'X' else 'X'

    while not game.game_over():
        game.print_board()
        if game.current == human:
            while True:
                try:
                    pos = input(f"Sua jogada ({human}), escolha linha e coluna (0-{ROWS-1})(0-{COLS-1}): ").strip().split()
                    if len(pos) != 2:
                        print("Digite somente 2 números separados por espaço")
                        continue
                    row, col = int(pos[0]), int(pos[1])

                    if (row, col) in game.available_moves():
                        game.make_move((row,col))
                        time.sleep(0.5)
                        break
                    else:
                        print("Posição inválida ou ocupada.")
                except ValueError:
                    print("Entrada inválida.")
        else:
            print("IA pensando...")
            if algorithm == "mcts":
                move = mcts(game, iterations=200)
            elif algorithm == "minimax":
                move = best_move_minimax(game)
            else:
                raise ValueError(f"Algoritmo desconhecido: {algorithm}")
            print(f"IA joga na coluna {move}")
            game.make_move(move)
            time.sleep(0.8)

    game.print_board()
    winner = game.winner()
    if winner == human:
        print(Fore.GREEN + "Você venceu!")
    elif winner == ai:
        print(Fore.RED + "A IA venceu.")
    else:
        print(Fore.CYAN + "Empate.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Jogo Gomoku com IA usando MCTS ou Minimax."
    )
    parser.add_argument(
        "--algo", "-a",
        choices=["mcts", "minimax"],
        required=True,
        help="Algoritmo da IA a ser usado (mcts ou minimax)."
    )
    args = parser.parse_args()
    play(args.algo)
