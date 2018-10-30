#include <iostream>
#include <cstdio>
#include <queue>
#include <cstring>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <algorithm>
// DWO
#define DWO false
// The type of constraint
#define ROWNOTEQUAL 0
#define COLNOTEQUAL 1
#define SMALL 2
#define BIG 3

using namespace std;

// board
static int board[9][9] = { { 0, 0, 0, 0, 0, 0, 0, 0, 0 },
{ 0, 0, 0, 0, 0, 0, 0, 0, 0 },
{ 0, 0, 0, 0, 0, 0, 0, 0, 0 },
{ 0, 0, 0, 0, 0, 0, 0, 0, 0 },
{ 0, 0, 0, 0, 0, 0, 0, 0, 0 },
{ 0, 0, 0, 0, 0, 0, 0, 0, 0 },
{ 0, 0, 0, 0, 0, 0, 0, 0, 0 },
{ 0, 0, 0, 0, 0, 0, 0, 0, 0 },
{ 0, 0, 0, 0, 0, 0, 0, 0, 0 } };
// state[i * 9 + j][x * 9 + y] = -1 means board[i][j] 
// must be smaller than board[x][y];
// state[i * 9 + j][x * 9 + y] = -1 means board[i][j]
// must be bigger than board[x][y];
static int state_compare[81][81];
// the size of board
static int board_size = 0;
// steps
static const int row_steps[4] = { 0, -1, 0, 1 }, col_steps[4] = { -1, 0, 1, 0 };
static bool isvis_compare[81][81];

// The state of a position
struct Node {
	int row, col;
	int able_put_down[9];
	int states_num;
	void initAblePutDown() {
		states_num = board_size;
		for (int i = 0; i < board_size; ++i) {
			able_put_down[i] = 1;
		}
	}
	Node(int row = 0, int col = 0) :row(row), col(col), states_num(0) {}
	void operator = (const Node &next) {
		row = next.row;
		col = next.col;
		states_num = next.states_num;
		for (int i = 0; i < board_size; ++i) {
			able_put_down[i] = next.able_put_down[i];
		}
	}
}node_board[9][9];

// Constraint class
struct Constraint {
	int type;
	pair<int, int> left, right;
	Constraint(const pair<int, int>& left, int type = ROWNOTEQUAL)
		:left(left), type(type) {}
	Constraint(const pair<int, int> &left, const pair<int, int> &right, int type = SMALL)
		:left(left), right(right), type(type) {}
	bool judge(const int &left_num, const int &right_num) const {
		if (type == ROWNOTEQUAL)return left_num != right_num;
		if (type == COLNOTEQUAL)return left_num != right_num;
		if (type == BIG)return left_num > right_num;
		if (type == SMALL)return left_num < right_num;
		return false;
	}
};

static vector<Constraint> all_constraint[81];
// Compare struct
struct myCmp {
	bool operator()(Node *a, Node *b) {
		if (a->states_num == b->states_num) {
			if (a->row > b->row)return true;
			else if (a->row == b->row) {
				return a->col > b->col;
			}
		}
		// sorted by states_num
		return a->states_num > b->states_num;
	}
}obj;

// find another variable that satisfies the Constraint
bool findCanPutDownPos(queue<Constraint> &gac_queue, const Constraint &now_con, Node &now_node, Node &next_node, const pair<int, int> &next_pair, bool is_init = 0) {
	int flag;
	int next_row = next_pair.first, next_col = next_pair.second, next_pos = next_row * board_size + next_col;
	for (int d = 0; d < board_size; ++d) {
		if (now_node.able_put_down[d] <= 0)continue;
		flag = 0;

		for (int nd = 0; nd < board_size; ++nd) {
			if (next_node.able_put_down[nd] <= 0)continue;
			if (now_con.judge(d, nd)) {
				flag = 1;
				break;
			}
		}

		if (!flag) {
			now_node.able_put_down[d]--;
			now_node.states_num--;
			if (now_node.states_num <= 0) {
				return DWO;
			}
		}
	}

	for (int d = 0; d < board_size; ++d) {
		if (next_node.able_put_down[d] <= 0)continue;
		flag = 0;

		for (int nd = 0; nd < board_size; ++nd) {
			if (now_node.able_put_down[nd] <= 0)continue;
			if (now_con.judge(nd, d)) {
				flag = 1;
				break;
			}
		}

		if (!flag) {
			next_node.able_put_down[d]--;
			next_node.states_num--;
			if (next_node.states_num <= 0) {
				return DWO;
			}
			if (!is_init) {
				gac_queue.push(all_constraint[next_pos][0]);
				gac_queue.push(all_constraint[next_pos][1]);
				int n_row, n_col, n_pos;
				for (int j = 2; j < all_constraint[next_pos].size(); ++j) {
					n_row = all_constraint[next_pos][j].right.first;
					n_col = all_constraint[next_pos][j].right.second;
					n_pos = n_row * board_size + n_col;
					if (board[n_row][n_col] || isvis_compare[next_pos][n_pos] || isvis_compare[n_pos][next_pos])continue;
					gac_queue.push(all_constraint[next_pos][j]);
					isvis_compare[next_pos][n_pos] = 1;
					isvis_compare[n_pos][next_pos] = 1;
				}
			}
		}
	}
	return true;
}

// init the board
inline void init() {
	int now_pos, next_pos, next_row, next_col;
	for (int i = 0; i < board_size; ++i) {
		for (int j = 0; j < board_size; ++j) {
			node_board[i][j].row = i;
			node_board[i][j].col = j;
			now_pos = i * board_size + j;
			pair<int, int> now_pair(i, j);
			if (board[i][j] > 0) {
				memset(node_board[i][j].able_put_down, 0, sizeof(node_board[i][j].able_put_down));
				node_board[i][j].able_put_down[board[i][j] - 1] = 1;
				node_board[i][j].states_num = 1;
			}

			all_constraint[now_pos].push_back(Constraint(now_pair, ROWNOTEQUAL));
			all_constraint[now_pos].push_back(Constraint(now_pair, COLNOTEQUAL));

			for (int k = 0; k < 4; ++k) {
				next_row = i + row_steps[k];
				next_col = j + col_steps[k];
				next_pos = next_row * board_size + next_col;
				if (state_compare[now_pos][next_pos] == -1) {
					all_constraint[i * board_size + j].push_back(Constraint(now_pair, make_pair(next_row, next_col), SMALL));
				}
				else if (state_compare[now_pos][next_pos] == 1) {
					all_constraint[i * board_size + j].push_back(Constraint(now_pair, make_pair(next_row, next_col), BIG));
				}
			}
		}
	}
}

bool GACQueue(queue<Constraint> &gac_queue, Node now_board[9][9], vector<Node*> &temp_queue, bool is_init = 0) {
	int now_row, now_col, next_row = 0, next_col = 0, now_pos = 0, next_pos = 0;

	bool flag = 0, isvis[9][9];
	for (int i = 0; i < board_size * board_size; ++i) {
		for (int j = 0; j < board_size * board_size; ++j) {
			if (i < board_size && j < board_size) {
				isvis[i][j] = 0;
			}
			isvis_compare[i][j] = isvis_compare[i][j] = 0;
		}
	}

	while (!gac_queue.empty()) {
		Constraint now_con = gac_queue.front();
		pair<int, int> now_pair = now_con.left;
		now_row = now_pair.first;
		now_col = now_pair.second;
		isvis[now_row][now_col] = 1;
		isvis[now_row][now_col] = 1;

		if (now_con.type <= 1) {
			for (int i = 0; i < board_size; ++i) {
				if (now_con.type == ROWNOTEQUAL) {
					next_row = now_row;
					next_col = i;
				}
				else {
					next_row = i;
					next_col = now_col;
				}
				if ((!is_init && board[next_row][next_col]) || isvis[next_row][next_col])continue;

				flag = findCanPutDownPos(gac_queue, now_con, now_board[now_row][now_col], now_board[next_row][next_col], make_pair(next_row, next_col), is_init);
				if (flag == DWO) {
					return DWO;
				}
			}
		}
		else {
			next_row = now_con.right.first;
			next_col = now_con.right.second;
			if ((!is_init && board[next_row][next_col]) || isvis[next_row][next_col]) {
				gac_queue.pop();
				continue;
			}

			flag = findCanPutDownPos(gac_queue, now_con, now_board[now_row][now_col], now_board[next_row][next_col], make_pair(next_row, next_col), is_init);
			if (flag == DWO) {
				return DWO;
			}
		}
		gac_queue.pop();
	}
	if(!temp_queue.empty())sort(temp_queue.begin(), temp_queue.end(), obj);
	return true;
}

// the first-time GACQUEUE()
inline void GACInit() {
	queue<Constraint> gac_queue;
	for (int i = 0; i < board_size * board_size; ++i) {
		for (auto x : all_constraint[i])gac_queue.push(x);
	}
	vector<Node*> temp_queue;
	GACQueue(gac_queue, node_board, temp_queue, 1);
}

bool GAC(Node node_board[9][9], vector<Node*> &search_queue, int depth) {
	if (search_queue.size() == 0) {
		return true;
	}
	Node* now_node = search_queue.back();
	if (now_node->states_num <= 0) {
		return false;
	}

	int now_row = now_node->row, now_col = now_node->col, next_row = 0, next_col = 0, now_pos = now_row * board_size + now_col;
	pair<int, int> now_pair = make_pair(now_row, now_col);
	bool flag = false;
	Node temp_board[9][9];
	vector<Node*> temp_queue;
	for (int i = 0; i < search_queue.size(); ++i) {
		temp_queue.push_back(&temp_board[search_queue[i]->row][search_queue[i]->col]);
	}

	for (int num = 1; num <= board_size; ++num) {
		if (node_board[now_row][now_col].able_put_down[num - 1] > 0) {
			// "put down" num on board[now_row][now_col]
			board[now_row][now_col] = num;
			for (int i = 0; i < board_size; ++i) {
				for (int j = 0; j < board_size; ++j) {
					temp_board[i][j] = node_board[i][j];
				}
			}
			temp_queue.pop_back();

			queue<Constraint> gac_queue;
			// push all constraint relevant to the variable into the queue
			gac_queue.push(all_constraint[now_pos][0]);
			gac_queue.push(all_constraint[now_pos][1]);
			int n_row, n_col, n_pos;
			for (int j = 2; j < all_constraint[now_pos].size(); ++j) {
				n_row = all_constraint[now_pos][j].right.first;
				n_col = all_constraint[now_pos][j].right.second;
				n_pos = n_row * board_size + n_col;
				if (board[n_row][n_col] || isvis_compare[now_pos][n_pos] || isvis_compare[n_pos][now_pos])continue;
				gac_queue.push(all_constraint[now_pos][j]);
			}
			memset(temp_board[now_row][now_col].able_put_down, 0, sizeof(int) * board_size);
			temp_board[now_row][now_col].able_put_down[num - 1] = 1;
			temp_board[now_row][now_col].states_num = 1;

			if (GACQueue(gac_queue, temp_board, temp_queue)) {
				flag = GAC(temp_board, temp_queue, depth + 1);
				if (flag)break;
			}
			// "pick up" num on board[now_row][now_col]
			temp_queue.push_back(&temp_board[now_row][now_col]);
			board[now_row][now_col] = 0;
		}
	}
	return flag;
}

// init the constraints and board
inline void Test1() {
	board_size = 4;
	board[0][2] = 3;
	for (int i = 0; i < board_size; ++i) {
		for (int j = 0; j < board_size; ++j) {
			node_board[i][j].initAblePutDown();
		}
	}

	state_compare[1][5] = -1;
	state_compare[5][1] = 1;
	state_compare[2][3] = 1;
	state_compare[3][2] = -1;
	state_compare[6][10] = -1;
	state_compare[10][6] = 1;
	state_compare[12][13] = 1;
	state_compare[13][12] = -1;
	state_compare[13][14] = 1;
	state_compare[14][13] = -1;

	init();
}

inline void Test2() {
	board_size = 5;
	board[4][4] = 4;
	for (int i = 0; i < board_size; ++i) {
		for (int j = 0; j < board_size; ++j) {
			node_board[i][j].initAblePutDown();
		}
	}

	state_compare[0][1] = 1;
	state_compare[1][0] = -1;
	state_compare[0][5] = -1;
	state_compare[5][0] = 1;
	state_compare[6][7] = -1;
	state_compare[7][6] = 1;
	state_compare[7][8] = -1;
	state_compare[8][7] = 1;
	state_compare[8][9] = -1;
	state_compare[9][8] = 1;
	state_compare[11][12] = -1;
	state_compare[12][11] = 1;
	state_compare[20][21] = 1;
	state_compare[21][20] = -1;

	init();
}

inline void Test3() {
	board_size = 6;
	board[0][4] = 2;
	board[0][5] = 6;
	board[1][5] = 3;
	board[2][0] = 3;
	board[3][2] = 4;
	for (int i = 0; i < board_size; ++i) {
		for (int j = 0; j < board_size; ++j) {
			node_board[i][j].initAblePutDown();
		}
	}

	state_compare[0][1] = 1;
	state_compare[1][0] = -1;
	state_compare[7][13] = 1;
	state_compare[13][7] = -1;
	state_compare[9][10] = 1;
	state_compare[10][9] = -1;
	state_compare[11][17] = -1;
	state_compare[17][11] = 1;
	state_compare[12][13] = -1;
	state_compare[13][12] = 1;
	state_compare[20][21] = 1;
	state_compare[21][20] = -1;
	state_compare[21][22] = 1;
	state_compare[22][21] = -1;
	state_compare[33][34] = -1;
	state_compare[34][33] = 1;
	state_compare[34][35] = -1;
	state_compare[35][34] = 1;

	init();
}

inline void Test4() {
	board_size = 7;
	board[0][6] = 6;
	board[3][6] = 2;
	board[5][1] = 5;
	for (int i = 0; i < board_size; ++i) {
		for (int j = 0; j < board_size; ++j) {
			node_board[i][j].initAblePutDown();
		}
	}

	state_compare[0][1] = -1;
	state_compare[1][0] = 1;
	state_compare[1][2] = 1;
	state_compare[2][1] = -1;
	state_compare[5][6] = 1;
	state_compare[6][5] = -1;
	state_compare[9][16] = 1;
	state_compare[16][9] = -1;
	state_compare[11][12] = 1;
	state_compare[12][11] = -1;
	state_compare[14][15] = -1;
	state_compare[15][14] = 1;
	state_compare[15][16] = -1;
	state_compare[16][15] = 1;
	state_compare[18][25] = -1;
	state_compare[25][18] = 1;
	state_compare[19][20] = 1;
	state_compare[20][19] = -1;
	state_compare[22][23] = -1;
	state_compare[23][22] = 1;
	state_compare[22][29] = 1;
	state_compare[29][22] = -1;
	state_compare[24][31] = -1;
	state_compare[31][24] = 1;
	state_compare[25][26] = 1;
	state_compare[26][25] = -1;
	state_compare[29][30] = 1;
	state_compare[30][29] = -1;
	state_compare[30][37] = 1;
	state_compare[37][30] = -1;
	state_compare[35][36] = -1;
	state_compare[36][35] = 1;
	state_compare[39][46] = -1;
	state_compare[46][39] = 1;
	state_compare[40][47] = 1;
	state_compare[47][40] = -1;
	state_compare[47][48] = 1;
	state_compare[48][47] = -1;

	init();
}

inline void Test5() {
	board_size = 8;
	board[1][4] = 6;
	board[1][6] = 7;
	board[2][3] = 4;
	board[4][7] = 6;
	board[5][5] = 4;
	board[6][7] = 3;
	for (int i = 0; i < board_size; ++i) {
		for (int j = 0; j < board_size; ++j) {
			node_board[i][j].initAblePutDown();
		}
	}

	state_compare[1][2] = 1;
	state_compare[2][1] = -1;
	state_compare[2][3] = 1;
	state_compare[3][2] = -1;
	state_compare[4][5] = -1;
	state_compare[5][4] = 1;
	state_compare[5][6] = -1;
	state_compare[6][5] = 1;
	state_compare[6][7] = -1;
	state_compare[7][6] = 1;
	state_compare[8][9] = -1;
	state_compare[9][8] = 1;
	state_compare[10][18] = -1;
	state_compare[18][10] = 1;
	state_compare[11][19] = -1;
	state_compare[19][11] = 1;
	state_compare[13][14] = -1;
	state_compare[14][13] = 1;
	state_compare[13][21] = 1;
	state_compare[21][13] = -1;
	state_compare[14][22] = 1;
	state_compare[22][14] = -1;
	state_compare[18][19] = -1;
	state_compare[19][18] = 1;
	state_compare[19][27] = 1;
	state_compare[27][19] = -1;
	state_compare[24][25] = 1;
	state_compare[25][24] = -1;
	state_compare[25][26] = 1;
	state_compare[26][25] = -1;
	state_compare[27][35] = 1;
	state_compare[35][27] = -1;
	state_compare[31][39] = 1;
	state_compare[39][31] = -1;
	state_compare[32][33] = 1;
	state_compare[33][32] = -1;
	state_compare[37][45] = -1;
	state_compare[45][37] = 1;
	state_compare[38][39] = -1;
	state_compare[39][38] = 1;
	state_compare[39][47] = -1;
	state_compare[47][39] = 1;
	state_compare[44][45] = -1;
	state_compare[45][44] = 1;
	state_compare[44][52] = 1;
	state_compare[52][44] = -1;
	state_compare[45][46] = 1;
	state_compare[46][45] = -1;
	state_compare[50][51] = 1;
	state_compare[51][50] = -1;

	init();
}

inline void Test6() {
	board_size = 9;
	board[0][3] = 7;
	board[0][4] = 3;
	board[0][5] = 8;
	board[0][7] = 5;
	board[1][2] = 7;
	board[1][5] = 2;
	board[2][5] = 9;
	board[3][3] = 4;
	board[4][2] = 1;
	board[4][6] = 6;
	board[4][7] = 4;
	board[5][6] = 2;
	board[8][8] = 6;
	for (int i = 0; i < board_size; ++i) {
		for (int j = 0; j < board_size; ++j) {
			node_board[i][j].initAblePutDown();
		}
	}

	// init the relation between the state
	state_compare[0][1] = -1;
	state_compare[1][0] = 1;
	state_compare[2][3] = 1;
	state_compare[3][2] = -1;
	state_compare[12][13] = -1;
	state_compare[13][12] = 1;
	state_compare[15][16] = -1;
	state_compare[16][15] = 1;
	state_compare[15][24] = 1;
	state_compare[24][15] = -1;
	state_compare[18][19] = 1;
	state_compare[19][18] = -1;
	state_compare[20][21] = -1;
	state_compare[21][20] = 1;
	state_compare[21][30] = -1;
	state_compare[30][21] = 1;
	state_compare[28][37] = 1;
	state_compare[37][28] = -1;
	state_compare[29][30] = 1;
	state_compare[30][29] = -1;
	state_compare[31][32] = 1;
	state_compare[32][31] = -1;
	state_compare[32][41] = 1;
	state_compare[41][32] = -1;
	state_compare[32][33] = -1;
	state_compare[33][32] = 1;
	state_compare[34][35] = 1;
	state_compare[35][34] = -1;
	state_compare[36][37] = -1;
	state_compare[37][36] = 1;
	state_compare[40][49] = 1;
	state_compare[49][40] = -1;
	state_compare[44][53] = 1;
	state_compare[53][44] = -1;
	state_compare[46][55] = -1;
	state_compare[55][46] = 1;
	state_compare[46][47] = -1;
	state_compare[47][46] = 1;
	state_compare[49][50] = -1;
	state_compare[50][49] = 1;
	state_compare[51][52] = 1;
	state_compare[52][51] = -1;
	state_compare[51][60] = 1;
	state_compare[60][51] = -1;
	state_compare[53][62] = 1;
	state_compare[62][53] = -1;
	state_compare[57][58] = -1;
	state_compare[58][57] = 1;
	state_compare[61][70] = 1;
	state_compare[70][61] = -1;
	state_compare[64][73] = -1;
	state_compare[73][64] = 1;
	state_compare[65][74] = 1;
	state_compare[74][65] = -1;
	state_compare[68][77] = -1;
	state_compare[77][68] = 1;
	state_compare[71][80] = 1;
	state_compare[80][71] = -1;
	state_compare[77][78] = -1;
	state_compare[78][77] = 1;
	// init the board
	init();
}

void TestPrepare(int i) {
	memset(board, 0, sizeof(board));
	memset(state_compare, 0, sizeof(state_compare));
	if (i == 1)Test1();
	else if (i == 2)Test2();
	else if (i == 3)Test3();
	else if (i == 4)Test4();
	else if (i == 5)Test5();
	else if (i == 6)Test6();
}

void Test(vector<Node*> &search_queue, int choice = 1) {
	for (int i = 0; i < board_size; ++i) {
		for (int j = 0; j < board_size; ++j) {
			if (board[i][j] == 0) {
				search_queue.push_back(&node_board[i][j]);
			}
		}
	}
	sort(search_queue.begin(), search_queue.end(), obj);
	if (GAC(node_board, search_queue, 0)) {
		for (int i = 0; i < board_size; ++i) {
			for (int j = 0; j < board_size; ++j) {
				printf("%d ", board[i][j]);
			}
			printf("\n");
		}
	}
	else {
		printf("False\n");
	}
	for (int i = 0; i < board_size * board_size; ++i) {
		while (all_constraint[i].size())all_constraint[i].pop_back();
	}
}

int main() {
	for (int i = 1; i <= 6; ++i) {
		printf("Test %d:\n", i);
		vector<Node*> search_queue;
		TestPrepare(i);
		double t = clock();
		GACInit();
		Test(search_queue, 1);
		printf("Time: %lfs\n", (clock() - t) / CLOCKS_PER_SEC);
	}
	system("pause");
	return 0;
}
