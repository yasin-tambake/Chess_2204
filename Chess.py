import math
import random
import time

# --- CORE GAME BOARD AND STATE MANAGEMENT ---

class ChessEngine:
    """
    Handles the game state, move validation, and board manipulation.
    White pieces are uppercase (e.g., 'P', 'R'), Black are lowercase (e.g., 'p', 'r').
    """
    def __init__(self):
        # 8x8 board representation. A simple 2D list.
        self.board = [
            ['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r'],
            ['p', 'p', 'p', 'p', 'p', 'p', 'p', 'p'],
            [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
            [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
            [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
            [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
            ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],
            ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R']
        ]
        self.white_to_move = True
        self.move_log = []
        self.white_king_pos = (7, 4)
        self.black_king_pos = (0, 4)
        self.checkmate = False
        self.stalemate = False

    def make_move(self, move):
        """Executes a move object (start_sq, end_sq)."""
        r1, c1 = move.start_sq
        r2, c2 = move.end_sq
        piece = self.board[r1][c1]

        # 1. Update the board
        self.board[r2][c2] = piece
        self.board[r1][c1] = ' '

        # 2. Update king positions
        if piece == 'K':
            self.white_king_pos = (r2, c2)
        elif piece == 'k':
            self.black_king_pos = (r2, c2)

        # 3. Handle Pawn Promotion (simplification: auto-promote to Queen)
        if piece == 'P' and r2 == 0:
            self.board[r2][c2] = 'Q'
        elif piece == 'p' and r2 == 7:
            self.board[r2][c2] = 'q'

        # 4. Log and change turn
        self.move_log.append(move)
        self.white_to_move = not self.white_to_move

    def undo_move(self):
        """Undoes the last move (used by Minimax)."""
        if not self.move_log:
            return

        move = self.move_log.pop()
        r1, c1 = move.start_sq
        r2, c2 = move.end_sq

        # 1. Restore the board
        self.board[r1][c1] = self.board[r2][c2] # Moving piece back
        self.board[r2][c2] = move.captured_piece # Restoring captured piece

        # 2. Update king positions if needed
        if self.board[r1][c1] == 'K':
            self.white_king_pos = (r1, c1)
        elif self.board[r1][c1] == 'k':
            self.black_king_pos = (r1, c1)

        # 3. Change turn back
        self.white_to_move = not self.white_to_move
        self.checkmate = False
        self.stalemate = False


    def get_valid_moves(self):
        """
        Gets all moves where the King is not left in check.
        This is the most computationally expensive part.
        """
        # 1. Get all pseudo-legal moves for the current player
        moves = self.get_all_possible_moves(self.white_to_move)

        # 2. Filter moves that result in a checked King
        legal_moves = []
        for move in moves:
            self.make_move(move) # Make the move
            
            # --- START FIX ---
            # self.make_move() flips the turn. We need to check the king of the
            # player who *just moved*, so we must flip the turn back temporarily.
            self.white_to_move = not self.white_to_move
            
            if not self.is_in_check():
                legal_moves.append(move)
                
            # Flip the turn *forward again* so undo_move() works correctly
            self.white_to_move = not self.white_to_move
            # --- END FIX ---
            
            self.undo_move() # Undo the move

        # 3. Check for Checkmate or Stalemate
        if not legal_moves:
            if self.is_in_check():
                self.checkmate = True
            else:
                self.stalemate = True
        else:
            self.checkmate = False
            self.stalemate = False

        return legal_moves

    def is_in_check(self):
        """
        Checks if the King of the player whose turn it is now (i.e., the player
        who just moved, or the player whose turn it is *after* make_move/undo_move)
        is under attack.
        """
        # The player whose King we are checking is the player whose turn it currently is (self.white_to_move)
        is_king_white = self.white_to_move
        
        if is_king_white:
            return self.square_is_attacked(self.white_king_pos[0], self.white_king_pos[1], True)
        else:
            return self.square_is_attacked(self.black_king_pos[0], self.black_king_pos[1], False)

    def square_is_attacked(self, r, c, is_king_white):
        """
        Checks if a given square (r, c) is attacked by the OPPOSITE player, 
        explicitly specifying the attacking color.
        """
        # Opponent color is the inverse of the king color
        opp_is_white = not is_king_white
        
        # Generate moves for the opponent's color
        opp_moves = self.get_all_possible_moves(opp_is_white) 

        for move in opp_moves:
            if move.end_sq == (r, c):
                return True
        return False

    def get_all_possible_moves(self, current_player_is_white):
        """Generates all pseudo-legal moves for the player specified by current_player_is_white."""
        moves = []
        for r in range(8):
            for c in range(8):
                piece = self.board[r][c]
                turn = current_player_is_white
                if (turn and 'A' <= piece <= 'Z') or (not turn and 'a' <= piece <= 'z'):
                    # Call appropriate move generator, passing the turn explicitly
                    if piece in ('p', 'P'): self._get_pawn_moves(r, c, moves, current_player_is_white)
                    elif piece in ('r', 'R'): self._get_rook_moves(r, c, moves, current_player_is_white)
                    elif piece in ('n', 'N'): self._get_knight_moves(r, c, moves, current_player_is_white)
                    elif piece in ('b', 'B'): self._get_bishop_moves(r, c, moves, current_player_is_white)
                    elif piece in ('q', 'Q'): self._get_queen_moves(r, c, moves, current_player_is_white)
                    elif piece in ('k', 'K'): self._get_king_moves(r, c, moves, current_player_is_white)
        return moves

    # --- PIECE MOVEMENT HELPERS ---
    # ALL helpers now accept 'is_white' instead of relying on self.white_to_move

    def _get_pawn_moves(self, r, c, moves, is_white):
        """Generates pawn moves (forward, initial double, captures)."""
        direction = -1 if is_white else 1
        start_row = 6 if is_white else 1
        opp_color = 'a' if is_white else 'A' # Opponent's first letter case

        # 1. Single square forward
        if 0 <= r + direction < 8 and self.board[r + direction][c] == ' ':
            moves.append(Move((r, c), (r + direction, c), self.board))
            # 2. Double square forward
            if r == start_row and self.board[r + 2 * direction][c] == ' ':
                moves.append(Move((r, c), (r + 2 * direction, c), self.board))

        # 3. Captures (left and right)
        for dc in [-1, 1]:
            if 0 <= c + dc < 8 and 0 <= r + direction < 8:
                target_piece = self.board[r + direction][c + dc]
                if target_piece != ' ' and (opp_color <= target_piece <= (chr(ord(opp_color) + 25))):
                    moves.append(Move((r, c), (r + direction, c + dc), self.board))

    def _get_rook_moves(self, r, c, moves, is_white):
        """Generates moves for Rooks and Queens (along straight lines)."""
        self._get_straight_line_moves(r, c, moves, [(0, 1), (0, -1), (1, 0), (-1, 0)], is_white)

    def _get_bishop_moves(self, r, c, moves, is_white):
        """Generates moves for Bishops and Queens (along diagonals)."""
        self._get_straight_line_moves(r, c, moves, [(1, 1), (1, -1), (-1, 1), (-1, -1)], is_white)

    def _get_queen_moves(self, r, c, moves, is_white):
        """Generates moves for Queens (straight and diagonal)."""
        self._get_straight_line_moves(r, c, moves, [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)], is_white)

    def _get_straight_line_moves(self, r, c, moves, directions, is_white):
        """Helper for Rook, Bishop, Queen moves."""
        my_piece = 'A' if is_white else 'a'
        opp_piece = 'a' if is_white else 'A'

        for dr, dc in directions:
            for i in range(1, 8):
                r_end, c_end = r + dr * i, c + dc * i
                if 0 <= r_end < 8 and 0 <= c_end < 8:
                    end_piece = self.board[r_end][c_end]
                    if end_piece == ' ': # Empty square
                        moves.append(Move((r, c), (r_end, c_end), self.board))
                    elif (opp_piece <= end_piece <= (chr(ord(opp_piece) + 25))): # Capture
                        moves.append(Move((r, c), (r_end, c_end), self.board))
                        break # Stop after capture
                    else: # Friendly piece
                        break
                else: # Off board
                    break

    def _get_knight_moves(self, r, c, moves, is_white):
        """Generates moves for Knights."""
        knight_moves = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
        opp_piece = 'a' if is_white else 'A'

        for dr, dc in knight_moves:
            r_end, c_end = r + dr, c + dc
            if 0 <= r_end < 8 and 0 <= c_end < 8:
                end_piece = self.board[r_end][c_end]
                if end_piece == ' ' or (opp_piece <= end_piece <= (chr(ord(opp_piece) + 25))):
                    moves.append(Move((r, c), (r_end, c_end), self.board))

    def _get_king_moves(self, r, c, moves, is_white):
        """Generates moves for Kings."""
        king_moves = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        opp_piece = 'a' if is_white else 'A'

        for dr, dc in king_moves:
            r_end, c_end = r + dr, c + dc
            if 0 <= r_end < 8 and 0 <= c_end < 8:
                end_piece = self.board[r_end][c_end]
                if end_piece == ' ' or (opp_piece <= end_piece <= (chr(ord(opp_piece) + 25))):
                    # Check if the move is legal (done later in get_valid_moves)
                    moves.append(Move((r, c), (r_end, c_end), self.board))


class Move:
    """Represents a move on the board."""
    ranks_to_rows = {"1": 7, "2": 6, "3": 5, "4": 4, "5": 3, "6": 2, "7": 1, "8": 0}
    rows_to_ranks = {v: k for k, v in ranks_to_rows.items()}
    files_to_cols = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7}
    cols_to_files = {v: k for k, v in files_to_cols.items()}

    def __init__(self, start_sq, end_sq, board):
        self.start_sq = start_sq  # (row, col)
        self.end_sq = end_sq      # (row, col)
        self.piece_moved = board[start_sq[0]][start_sq[1]]
        self.captured_piece = board[end_sq[0]][end_sq[1]]

    def get_chess_notation(self):
        """Converts move to standard notation (e.g., 'e7e8')."""
        return self.get_rank_file(self.start_sq) + self.get_rank_file(self.end_sq)

    def get_rank_file(self, sq):
        return self.cols_to_files[sq[1]] + self.rows_to_ranks[sq[0]]

    def __eq__(self, other):
        """Equality check for comparing moves."""
        if isinstance(other, Move):
            return self.start_sq == other.start_sq and self.end_sq == other.end_sq
        return False

# --- AI IMPLEMENTATION (MINIMAX WITH ALPHA-BETA PRUNING) ---

class ChessAI:
    """
    Implements the Minimax algorithm with Alpha-Beta Pruning.
    """
    PIECE_VALUES = {
        'P': 10, 'N': 30, 'B': 30, 'R': 50, 'Q': 90, 'K': 900,
        'p': -10, 'n': -30, 'b': -30, 'r': -50, 'q': -90, 'k': -900
    }
    # Bonus points for pieces being in central squares (optional, but improves AI)
    POSITION_BONUS = {
        (3, 3): 3, (3, 4): 3, (4, 3): 3, (4, 4): 3,
        (2, 2): 1, (2, 5): 1, (5, 2): 1, (5, 5): 1,
    }

    def __init__(self, depth=3):
        self.MAX_DEPTH = depth

    def get_best_move(self, gs):
        """Top-level function to start the minimax search."""
        moves = gs.get_valid_moves()
        random.shuffle(moves) # Randomize moves to break ties
        
        current_player_is_white = gs.white_to_move
        
        # Initialize best_score to the worst possible score for the current player
        # White (Maximizer) wants highest score (-inf is worst), Black (Minimizer) wants lowest score (+inf is worst).
        if current_player_is_white:
            best_score = -math.inf 
        else:
            best_score = math.inf 

        best_move = None
            
        start_time = time.time()
        print(f"\nAI (Depth {self.MAX_DEPTH}): Calculating move...")

        # Iterating over all moves is the most expensive step
        for move in moves:
            gs.make_move(move)
            
            # The next call to minimax is for the OPPONENT's turn
            score = self.minimax(gs, self.MAX_DEPTH - 1, -math.inf, math.inf, not current_player_is_white)
            
            if current_player_is_white: # If AI is White (Maximizer)
                 if score > best_score:
                    best_score = score
                    best_move = move
            else: # If AI is Black (Minimizer)
                # We need the lowest score for Black
                if score < best_score:
                    best_score = score
                    best_move = move

            gs.undo_move()

        end_time = time.time()
        print(f"Calculation complete in {end_time - start_time:.2f} seconds.")
        
        return best_move

    def minimax(self, gs, depth, alpha, beta, is_maximizing_player):
        """
        The Minimax algorithm with Alpha-Beta Pruning.
        Scores are calculated from White's perspective: +ve is good for White, -ve is good for Black.
        - True: White (Maximizing player)
        - False: Black (Minimizing player)
        """
        # 1. Base Case: Reached maximum depth or game is over
        if depth == 0 or gs.checkmate or gs.stalemate:
            return self.evaluate_board(gs)

        # 2. Recursive Step
        
        moves = gs.get_valid_moves()

        if is_maximizing_player: # White's turn (Maximizer)
            max_eval = -math.inf
            for move in moves:
                gs.make_move(move)
                eval = self.minimax(gs, depth - 1, alpha, beta, False)
                gs.undo_move()

                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break # Beta cutoff
            return max_eval

        else: # Black's turn (Minimizer)
            min_eval = math.inf
            for move in moves:
                gs.make_move(move)
                eval = self.minimax(gs, depth - 1, alpha, beta, True)
                gs.undo_move()

                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break # Alpha cutoff
            return min_eval

    def evaluate_board(self, gs):
        """
        Assigns a score to the current board state from White's perspective.
        +ve score favors White, -ve score favors Black.
        """
        if gs.checkmate:
            # If white to move and checkmate, black wins (low score)
            if gs.white_to_move:
                return -999999
            # If black to move and checkmate, white wins (high score)
            else:
                return 999999
        elif gs.stalemate:
            return 0 # Draw

        score = 0
        for r in range(8):
            for c in range(8):
                piece = gs.board[r][c]
                # Material Score
                score += self.PIECE_VALUES.get(piece, 0)
                
                # Positional Bonus
                if piece in ('P', 'N', 'B', 'R', 'Q'): # White pieces
                    score += self.POSITION_BONUS.get((r, c), 0)
                elif piece in ('p', 'n', 'b', 'r', 'q'): # Black pieces
                    # Note: Positional bonuses are inverted for Black
                    score -= self.POSITION_BONUS.get((r, c), 0)

        return score


# --- UI AND MAIN GAME LOOP ---

def print_board(board):
    """Prints the chessboard to the console with coordinates."""
    # Use ANSI codes for colors if the terminal supports them
    RESET = "\033[0m"
    WHITE_SQUARE = "\033[47m\033[30m" # White background, black text
    BLACK_SQUARE = "\033[40m\033[37m" # Black background, white text

    # --- FIX START ---
    # 3-space prefix to match the row prefix (e.g., "8 |")
    # 2 spaces between letters to match the 3-char width of a square
    print("   a  b  c  d  e  f  g  h") 
    # 3-space prefix + 24 hyphens (8 squares * 3 chars)
    print("   ------------------------") 
    # --- FIX END ---
    
    for r in range(8):
        row_str = f"{8 - r} |" # 3-char prefix
        for c in range(8):
            color = WHITE_SQUARE if (r + c) % 2 == 0 else BLACK_SQUARE
            piece = board[r][c]
            # Replace empty space with a subtle dot for better visibility
            display_piece = piece if piece != ' ' else '.'
            
            # Each square is 3 visible chars wide: " . " or " P "
            row_str += f"{color} {display_piece} {RESET}"
        print(row_str + f"| {8 - r}")
        
    # --- FIX START ---
    # Match the top border
    print("   ------------------------") 
    # Match the top file header
    print("   a  b  c  d  e  f  g  h")
    # --- FIX END ---

def handle_user_input(valid_moves, board):
    """
    Handles user input and returns a valid Move object.
    Now accepts the current game board to correctly initialize the Move object.
    """
    while True:
        try:
            user_input = input("Enter your move (e.g., 'e2e4'): ").strip().lower()

            if user_input == 'exit':
                return 'exit'

            if len(user_input) != 4:
                print("Invalid format. Use four characters (e.g., 'e2e4').")
                continue

            # Convert chess notation to row, col
            start_sq_str = user_input[:2]
            end_sq_str = user_input[2:]

            r1 = Move.ranks_to_rows[start_sq_str[1]]
            c1 = Move.files_to_cols[start_sq_str[0]]
            r2 = Move.ranks_to_rows[end_sq_str[1]]
            c2 = Move.files_to_cols[end_sq_str[0]]

            # Use the actual board now to initialize the move properly
            user_move = Move((r1, c1), (r2, c2), board) 
            
            # Find the actual move object from the valid list
            found_move = None
            for move in valid_moves:
                if move.start_sq == (r1, c1) and move.end_sq == (r2, c2):
                    found_move = move
                    break

            if found_move:
                return found_move
            else:
                print("Illegal move. Please try again.")

        except KeyError:
            print("Invalid square notation. Use standard notation (a1-h8).")
        except Exception as e:
            # Print the actual error for debugging, then prompt again
            print(f"An unexpected error occurred: {e}. Please try again.")
            
def main():
    """Main game execution loop."""
    print("--- Python Command-Line Chess (1P vs AI) ---")
    print("Your pieces are White (Uppercase). AI is Black (Lowercase).")
    print("Enter moves in notation like 'e2e4'. Type 'exit' to quit.")

    gs = ChessEngine()
    ai = ChessAI(depth=3) # Adjust depth here (3 is a good balance for speed/strength)

    print_board(gs.board)

    while not gs.checkmate and not gs.stalemate:
        player_turn = "White (You)" if gs.white_to_move else "Black (AI)"
        print(f"\n--- {player_turn}'s Turn ---")

        valid_moves = gs.get_valid_moves()

        if gs.white_to_move:
            # Player's turn
            # Pass the current board to the input handler
            move = handle_user_input(valid_moves, gs.board)
            if move == 'exit':
                break
        else:
            # AI's turn
            if not valid_moves:
                # Should be caught by get_valid_moves setting checkmate/stalemate, 
                # but this prevents the 'No valid moves found' message when it shouldn't happen.
                break 
            move = ai.get_best_move(gs)

        if move:
            gs.make_move(move)
            print(f"Executed move: {move.get_chess_notation()}")
            print_board(gs.board)
        else:
            print("No valid moves found. (Error state or end game check failed)")
            break

    # Game End Results
    if gs.stalemate:
        print("\n*** Stalemate! It's a Draw. ***")
    elif gs.checkmate:
        winner = "Black (AI)" if gs.white_to_move else "White (You)"
        print(f"\n*** Checkmate! {winner} wins! ***")
    else:
        print("\nGame aborted.")

if __name__ == "__main__":
    main()

