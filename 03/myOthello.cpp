#include <algorithm>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <windows.h>
#include <conio.h>
#include <cstdlib>
// chess color
#define BLACK 1
#define SPACE 0
#define WHITE -1
#define INFINITY 50000000
// search layer for different AI
#define LOW_MAX_STEP 6
#define HIGH_MAX_STEP 12

using namespace std;

// value of position
static const int pos_value[6][6] = { 
	{ 300, -40, 20, 20, -40, 300 },
	{ -40, -100, 5, 5, -100, -40 },
	{ 20, 5, 1, 1, 5, 20 },
	{ 20, 5, 1, 1, 5, 20 },
	{ -40, -100, 5, 5, -100, -40 },
	{ 300, -40, 20, 20, -40, 300 } };
// positions sorted by the value 
static pair<int, int> moves[36];
// position of the cursor
static int cursor_row = 0, cursor_col = 0;

// move the cursor on the cmd
void gotoxy(int x, int y) {
	int xx = 0x0b;
	HANDLE hOutput;
	COORD loc;
	loc.X = x;
	loc.Y = y;
	hOutput = GetStdHandle(STD_OUTPUT_HANDLE);
	SetConsoleCursorPosition(hOutput, loc);
	return;
}

// clear the cmd windows
void clearWindows() {
	gotoxy(0, 6);
	for (int i = 0; i < 16; ++i) {
		cout << "                             \n";
	}
}

struct Chess {
	int value; // the value of chess
	int color; // the color of chess
	Chess(int value = 0, int color = SPACE) 
		:value(value), color(color) {}
};

struct Choice {
	pair<int, int> pos; // record the position of next move
	int score; // record the score of next move
	Choice(int row = -1, int col = -1, int score = -1) 
		:pos(make_pair(row, col)), score(score) {}
};

struct ChessBoard {
	Chess board[6][6];
	int white_num;
	int black_num;

	ChessBoard() :white_num(2), black_num(2) {
		for (int i = 0; i < 6; ++i) {
			for (int j = 0; j < 6; ++j) {
				board[i][j] = Chess();
			}
		}
	}
	ChessBoard(const ChessBoard &now, int step = 1) 
		:white_num(now.white_num), black_num(now.black_num) {
		for (int i = 0; i < 6; ++i) {
			for (int j = 0; j < 6; ++j) {
				board[i][j].color = now.board[i][j].color;
				if(step == 1)board[i][j].value = now.board[i][j].value;
				else board[i][j].value = 0;
			}
		}
	}
	// prepare for creating a chessboard
	void Init(); 
	// calculate the total stable
	int calStable(int player); 
	// calculate the stable of a position
	int calPosStable(int now_row, int now_col, int player); 
	// calculate the value of player's opponent
	int calOpponentValue(int player);
	// calculate the value of a position
	void calPosValue(int now_row, int now_col, int player);
	int getVaildMove(pair<int, int> valid_moves[], int player);
	// change the color of the position(black or white)
	void changeColor(const pair<int, int>& move_pos, int color);
	// move the chess
	void chessOn(int player, const pair<int, int>& move_pos, 
		bool is_play = false);
	// player's moving function
	int playerMove(int player);
	// judge the game
	int Judge();
	// judge the position is valid or not
	static bool isVaildPos(int row, int col);
	// clear all the value of the chessboard
	void clearValue();
};

bool ChessBoard::isVaildPos(int row, int col) {
	return row >= 0 && row <= 5 && col >= 0 && col <= 5;
}

// The value of a position for player is the number of chess he can eat
// when he put his chess down here.
void ChessBoard::calPosValue(int now_row, int now_col, int player) {
	int next_row, next_col, line_value;
	// four directions
	for (int x = -1; x <= 1; ++x) {
		for (int y = -1; y <= 1; ++y) {
			if ((!x) && (!y))continue;
			next_row = now_row + x;
			next_col = now_col + y;
			line_value = 0;
			// calculate the value of one direction
			while (isVaildPos(next_row, next_col)) {
				if (board[next_row][next_col].color == -player) {
					++line_value;
				}
				else if (board[next_row][next_col].color == SPACE)break;
				else {
					board[now_row][now_col].value += line_value;
					break;
				}
				next_row += x;
				next_col += y;
			}
		}
	}
}

int ChessBoard::calOpponentValue(int player) {
	int value = 0;
	for (int i = 0; i < 5; i += 5) {
		for (int j = 0; j < 5; j += 5) {
			if (board[i][j].color != SPACE)continue;
			board[i][j].value = 0;
			calPosValue(i, j, -player);
			// it means player's opponent gets the corner.
			if (board[i][j].value > 0)value += 15 * board[i][j].value;
			board[i][j].value = 0;
		}
	}
	return -value;
}

int ChessBoard::getVaildMove(pair<int, int> valid_moves[], int player) {
	int moves_num = 0, now_row = 0, now_col = 0;
	// "moves" has been sorted by the value of position
	for (int k = 0; k < 36; ++k) {
		now_row = moves[k].first;
		now_col = moves[k].second;
		if (board[now_row][now_col].color == SPACE) {
			calPosValue(now_row, now_col, player);
		}
		// means the postion is valid
		if (board[now_row][now_col].value > 0) {
			valid_moves[moves_num++] = make_pair(now_row, now_col);
		}
	}
	return moves_num;
}

// The stable values the safety of this position of player
// If the player put chess down here and his opponent eat his chess at once,
// it means the position is not safe
int ChessBoard::calPosStable(int now_row, int now_col, int player) {
	int now_color = board[now_row][now_col].color;
	int pos_stable = 0, next_row = 0, next_col = 0, pos_line_stable = 2;
	bool is_left_space = false, is_right_space = false;
	if (now_color == SPACE)return 0;
	for (int x = -1; x <= 0; ++x) {
		for (int y = -1; y <= 0; ++y) {
			if (x == 0 && y == 0)break;
			pos_line_stable = 2;
			next_row = now_row + x;
			next_col = now_col + y;
			while (isVaildPos(next_row, next_col)) {
				if (board[next_row][next_col].color == -now_color) {
					pos_line_stable--;
					break;
				}
				if (board[next_row][next_col].color == SPACE) {
					is_left_space = true;
					break;
				}
				next_row += x;
				next_col += y;
			}
			next_row = now_row - x;
			next_col = now_col - y;
			while (isVaildPos(next_row, next_col)) {
				if (board[next_row][next_col].color == -now_color) {
					pos_line_stable--;
					break;
				}
				if (board[next_row][next_col].color == SPACE) {
					is_right_space = true;
					break;
				}
				next_row -= x;
				next_col -= y;
			}
			if (pos_line_stable == 1 && (is_left_space || is_right_space)) {
				pos_line_stable--;
			}
			pos_stable += pos_line_stable;
		}
	}
	return player == now_color ? pos_stable : -pos_stable;
}

int ChessBoard::calStable(int player) {
	int total_stable = 0;
	for (int i = 0; i < 6; ++i) {
		for (int j = 0; j < 6; ++j) {
			total_stable += 25 * calPosStable(i, j, player);
			if (board[i][j].color == SPACE)continue;
			if (player == board[i][j].color)total_stable += pos_value[i][j];
			else total_stable -= pos_value[i][j];
		}
	}
	return total_stable;
}

void ChessBoard::chessOn(int player, const pair<int, int>& move_pos, 
						bool is_play) {
	int now_row = move_pos.first, now_col = move_pos.second, next_row, next_col;
	if (board[now_row][now_col].color != SPACE || player == SPACE)return;
	white_num += (player == WHITE);
	black_num += (player == BLACK);
	for (int x = -1; x <= 1; ++x) {
		for (int y = -1; y <= 1; ++y) {
			if ((!x) && (!y))continue;
			next_row = now_row + x;
			next_col = now_col + y;
			while (isVaildPos(next_row, next_col)) {
				if (board[next_row][next_col].color == -player) {
					next_row += x;
					next_col += y;
					continue;
				}
				else if (board[next_row][next_col].color == SPACE)break;
				// eat opponent's chesses
				else {
					for (int i = 1; next_row - i * x != now_row 
						|| next_col - i * y != now_col; ++i) {
						board[next_row - i * x][next_col - i * y].color = player;
						white_num += ((player == WHITE) ? 1 : -1);
						black_num += ((player == BLACK) ? 1 : -1);
						if (is_play) {
							changeColor(make_pair(next_row - i * x, 
								next_col - i * y), player);
						}
					}
					break;
				}
				next_row += x;
				next_col += y;
			}
		}
	}
	board[now_row][now_col].color = player;
	if (is_play) {
		changeColor(make_pair(now_row, now_col), player);
	}
}

void ChessBoard::changeColor(const pair<int, int>& move_pos, int color) {
	gotoxy(move_pos.second * 2 + 1, move_pos.first);
	if (color == BLACK) {
		cout << "¡ð"; 
	}
	else {
		cout << "¡ñ";
	}
	gotoxy(cursor_row, cursor_col);
}

void ChessBoard::Init() {
	cursor_row = 1;
	cursor_col = 0;
	black_num = 2;
	white_num = 2;
	for (int i = 0; i < 6; ++i) {
		printf(" ");
		for (int j = 0; j < 6; ++j) {
			moves[i * 6 + j] = make_pair(i, j);
			if ((i == 2 && j == 2) || (i == 3 && j == 3)) {
				board[i][j].color = WHITE;
				printf("¡ñ"); 
			}
			else if ((i == 2 && j == 3) || (i == 3 && j == 2)) {
				board[i][j].color = BLACK;
				printf("¡ð"); 
			}
			else {
				board[i][j].color = SPACE;
				printf("+ ");
			}
		}
		printf("\n");
	}
	sort(moves, moves + 36,
		[&](const pair<int, int> &a, const pair<int, int> &b) {
		return pos_value[a.first][a.second] > pos_value[b.first][b.second];
	});
	gotoxy(cursor_row, cursor_col);
}

int ChessBoard::playerMove(int player) {
	char select = _getch();
	int now_row, now_col, flag = -1;
	switch (select) {
		// move
		case 'd': case 'D':
			if (cursor_row < 12) {
				cursor_row += 2;
				gotoxy(cursor_row, cursor_col);
				flag = 0;
			}
			break;
		case 'a': case 'A':
			if (cursor_row > 0) {
				cursor_row -= 2;
				gotoxy(cursor_row, cursor_col);
				flag = 0;
			}
			break;
		case 'w': case 'W':
			if (cursor_col > 0) {
				cursor_col -= 1;
				gotoxy(cursor_row, cursor_col);
				flag = 0;
			}
			break;
		case 's': case 'S':
			if (cursor_col < 6) {
				cursor_col += 1;
				gotoxy(cursor_row, cursor_col);
				flag = 0;
			}
			break;
		// put the chess down
		case 'j': case 'J':
			now_row = cursor_col;
			now_col = cursor_row / 2;
			if (board[now_row][now_col].color == SPACE) {
				calPosValue(now_row, now_col, player);
				if (board[now_row][now_col].value > 0) {
					chessOn(player, make_pair(now_row, now_col), true);
					flag = 1;
				}
			}
			break;
		default: break;
	}
	return flag;
}

int ChessBoard::Judge() {
	pair<int, int> black_valid_moves[36], white_valid_moves[36];
	clearValue();
	int black_moves = getVaildMove(black_valid_moves, BLACK);
	clearWindows();
	gotoxy(0, 6);
	cout << "Black(¡ð) num: " << black_num << "\n";
	clearValue();
	int white_moves = getVaildMove(white_valid_moves, WHITE);
	cout << "White(¡ñ) num: " << white_num << "\n";
	clearValue();
	gotoxy(cursor_row, cursor_col);
	// means game is over
	if (black_moves <= 0 && white_moves <= 0) {
		clearWindows();
		gotoxy(0, 6);
		cout << ((white_num > black_num) ? "WHITE(¡ñ) WINS" : 
			(white_num < black_num)? "BLACK(¡ð) WINS": "DRAW") << endl;
		return -2;
	}
	// means Black has no way to move
	else if (black_moves == 0 && white_moves > 0) {
		return WHITE;
	}
	// means White has no way to move
	else if (black_moves > 0 && white_moves == 0) {
		return BLACK;
	}
	return 0;
}

void ChessBoard::clearValue() {
	for (int i = 0; i < 6; ++i) {
		for (int j = 0; j < 6; ++j) {
			board[i][j].value = 0;
		}
	}
}

// alpha-beta
Choice alphaBeta(ChessBoard &now_board, int player, int step, 
				int alpha, int beta, pair<int, int> last_move) {
	Choice best_choice;
	pair<int, int> valid_moves[36];
	// get the valid moves for the player
	int moves_num = now_board.getVaildMove(valid_moves, player), 
		best = -INFINITY - 1;

	if (step <= 0) {
		int last_row = last_move.first, last_col = last_move.second;
		// calculate the score of this node
		int now_value = 0;

		now_value += now_board.calStable(player);
		now_value += now_board.calOpponentValue(player);

		now_value += 7 * moves_num;

		now_value += 5 * now_board.board[last_row][last_col].value;

		return Choice(last_row, last_col, now_value);
	}

	// it means the player has no position to move to
	if (!moves_num) {
		if (now_board.getVaildMove(valid_moves, -player) > 0) {
			ChessBoard temp_board(now_board);
			Choice next_choice = alphaBeta(temp_board, -player, 
				step - 1, -alpha, -beta, last_move);
			best_choice.pos.first = -1;
			best_choice.pos.second = -1;
			best_choice.score = -next_choice.score;
			return best_choice;
		}
		// it means the opponent also has no position to move to
		else {
			int value = WHITE * (now_board.white_num) + BLACK * (now_board.black_num);
			if (player*value > 0){
				best_choice.score = INFINITY - 1;
			}
			else if (player*value < 0){
				best_choice.score = -INFINITY + 1;
			}
			else{
				best_choice.score = 0;
			}
			return best_choice;
		}
	}

	// alpha-beta
	for (int i = 0; i < moves_num; ++i) {
		ChessBoard temp_board(now_board, step);
		temp_board.chessOn(player, valid_moves[i]);
		Choice next_choice = alphaBeta(temp_board, -player, 
			step - 1, -beta, -alpha, valid_moves[i]);
		if (-next_choice.score > best) {
			best_choice.pos.first = valid_moves[i].first;
			best_choice.pos.second = valid_moves[i].second;
			best = -next_choice.score;
			best_choice.score = best;
		}
		if (best > alpha)alpha = best;
		if (best >= beta)break;
	}
	return best_choice;
}

void playerVsAI(ChessBoard &new_chess_board) {
	system("cls");
	cout << "Othello V0.9\n1.Player hold black\n2.Player hold white\n";
	char select = _getch();
	int player, ai, player_flag, judge_flag;
	Choice next_choice;
	while (select != '1' && select != '2') {
		system("cls");
		cout << "ÊäÈë´íÎó\n";
		cout << "Othello V0.9\n1.Player hold black\n2.Player hold white\n";
		select = _getch();
	}
	if (select == '1')player = BLACK;
	else player = WHITE;
	ai = -player;
	cout << "\nWASD: Move the cursor\nJ: Put down the chess\n";
	system("pause");
	system("cls");
	new_chess_board.Init();
	for (int i = 0; i < 36; ++i) {
		player_flag = -1;
		judge_flag = new_chess_board.Judge();
		if (judge_flag == -2)break;
		if (i % 2 == 0) {
			if (select == '1' && judge_flag != WHITE) {
				while (player_flag <= 0) {
					player_flag = new_chess_board.playerMove(player);
				}
			}
			else {
				next_choice = alphaBeta(new_chess_board, ai, 
					HIGH_MAX_STEP, -INFINITY, INFINITY, make_pair(-1, -1));
				new_chess_board.chessOn(ai, next_choice.pos, true);
			}
		}
		else {
			if (select == '2' && judge_flag != BLACK) {
				while (player_flag <= 0) {
					player_flag = new_chess_board.playerMove(player);
				}
			}
			else {
				next_choice = alphaBeta(new_chess_board, ai, 
					HIGH_MAX_STEP, -INFINITY, INFINITY, make_pair(-1, -1));
				new_chess_board.chessOn(ai, next_choice.pos, true);
			}
		}
	}
	system("pause");
}

void AIVsAI(ChessBoard &new_chess_board) {
	system("cls");
	cout << "Othello V0.9\n1.Senior AI hold black\n2.Senior AI hold white\n";
	char select = _getch();
	int low_ai, high_ai, judge_flag;
	Choice next_choice;
	while (select != '1' && select != '2') {
		system("cls");
		cout << "Error\n";
		cout << "Othello V0.9\n1.Senior AI hold black\n2.Senior AI hold white\n";
		select = _getch();
	}
	system("cls");
	if (select == '1')high_ai = BLACK;
	else high_ai = WHITE;
	low_ai = -high_ai;
	new_chess_board.Init();
	for (int i = 0; i < 36; ++i) {
		judge_flag = new_chess_board.Judge();
		if (judge_flag == -2)break;
		if (i % 2 == 0) {
			if (select == '1' && judge_flag != WHITE) {
				next_choice = alphaBeta(new_chess_board, high_ai, 
					HIGH_MAX_STEP, -INFINITY, INFINITY, make_pair(-1, -1));
				new_chess_board.chessOn(high_ai, next_choice.pos, true);
				gotoxy(0, 9);
				cout << "Black Moves: (" << next_choice.pos.first 
					<< ", " << next_choice.pos.second << ")\n";
			}
			else {
				next_choice = alphaBeta(new_chess_board, low_ai, 
					LOW_MAX_STEP, -INFINITY, INFINITY, make_pair(-1, -1));
				new_chess_board.chessOn(low_ai, next_choice.pos, true);
				gotoxy(0, 9);
				cout << "White Moves: (" << next_choice.pos.first 
					<< ", " << next_choice.pos.second << ")\n";
			}
		}
		else {
			if (select == '2' && judge_flag != BLACK) {
				next_choice = alphaBeta(new_chess_board, high_ai, 
					HIGH_MAX_STEP, -INFINITY, INFINITY, make_pair(-1, -1));
				new_chess_board.chessOn(high_ai, next_choice.pos, true);
				gotoxy(0, 9);
				cout << "Black Moves: (" << next_choice.pos.first 
					<< ", " << next_choice.pos.second << ")\n";
			}
			else {
				next_choice = alphaBeta(new_chess_board, low_ai, 
					LOW_MAX_STEP, -INFINITY, INFINITY, make_pair(-1, -1));
				new_chess_board.chessOn(low_ai, next_choice.pos, true);
				gotoxy(0, 9);
				cout << "White Moves: (" << next_choice.pos.first 
					<< ", " << next_choice.pos.second << ")\n";
			}
		}
		gotoxy(next_choice.pos.second * 2 + 1, next_choice.pos.first);
		Sleep(1000);
	}
	system("pause");
}

int main() {
	ChessBoard new_chess_board;
	char choose;
	cout << "Othello V0.9\n1.Player VS AI\n2.AI VS AI\n";
	choose = _getch();
	while (1) {
		if (choose == '1') {
			system("pause");
			_getch();
			playerVsAI(new_chess_board);
		}
		else if (choose == '2') {
			system("pause");
			_getch();
			AIVsAI(new_chess_board);
		}
		system("cls");
		cout << "Othello V0.9\n1.Player VS AI\n2.AI VS AI\n";
		choose = _getch();
	}
	return 0;
}