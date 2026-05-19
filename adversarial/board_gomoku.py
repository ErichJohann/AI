from board_game import BoardGame
from helper_functions import colorize

ROWS, COLS = 15, 15

class Gomoku(BoardGame):
    def __init__(self):
        super().__init__(ROWS, COLS)
    
    def available_moves(self):
        moves = []
        for row in range(ROWS):
            for col in range(COLS):
                if self.board[row][col] == ' ':
                    moves.append((row, col))
        return moves

    def make_move(self, move):
        r, c = move
        if self.board[r][c] == ' ':
            self.board[r][c] = self.current
            self.current = 'O' if self.current == 'X' else 'X'
            return True
        return False
    
    def winner(self):
        for r in range(ROWS):
            for c in range(COLS):
                if self.board[r][c] == ' ':
                    continue

                if r + 4 < ROWS:
                    if all(self.board[r+i][c] == self.board[r][c] for i in range(5)):
                        return self.board[r][c]
                if c + 4 < COLS:
                    if all(self.board[r][c+i] == self.board[r][c] for i in range(5)):
                        return self.board[r][c]
                if r + 4 < ROWS and c + 4 < COLS:
                    if all(self.board[r+i][c+i] == self.board[r][c] for i in range(5)):
                        return self.board[r][c]
                if r + 4 < ROWS and c - 4 >= 0:
                    if all(self.board[r+i][c-i] == self.board[r][c] for i in range(5)):
                        return self.board[r][c]
        return None
    
    def print_board(self):
        for r in range(ROWS):
            # f"{r:2}" deixa os números 0-9 alinhados perfeitamente com os números 10-14
            row_str = f"{r:2} |"
            
            # Modifica as células para usar 2 espaços de largura para casar com o cabeçalho
            cells = []
            for cell in self.board[r]:
                if cell == ' ':
                    cells.append("  ")
                else:
                    cells.append(f" {colorize(cell)}")
                    
            row_str += "|".join(cells) + "|"
            print(row_str)

        print("   " + " ".join(f"{c:2}" for c in range(COLS)))