#include <cstdio>
#include <stack>
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <ctime>
#include <cstdlib>
#define INFINITY 10000000

using namespace std;

// target_rows[i] = j presents the number i is in jth row
const static int target_state[16] = { 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 };
const static int target_rows[16] = { 3, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3 };
const static int target_cols[16] = { 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2 };
// 1 means using the fast manhattan distance
int IS_FAST = 0; 

// the struct of state
struct State {
	int zero_pos; // the postion of zero
	int state[16]; 
	State() :zero_pos(15) {}
	State(int next_state[]) {
		for (int i = 0; i < 16; ++i) {
			state[i] = next_state[i];
			if (!state[i]) {
				zero_pos = i;
			}
		}
	}
	State(const State& next) {
		for (int i = 0; i < 16; ++i) {
			state[i] = next.state[i];
			if (!state[i]) {
				zero_pos = i;
			}
		}
	}
	void swap(int i, int j) {
		int temp = state[i];
		int next_zero_pos = (temp == 0) ? j : i;
		state[i] = state[j];
		state[j] = temp;
		zero_pos = next_zero_pos;
	}
	void setState(int next_state[]) {
		for (int i = 0; i < 16; ++i) {
			state[i] = next_state[i];
			if (!state[i]) {
				zero_pos = i;
			}
		}
	}
} total_state;

// calculate the manhattan distance
int calManhattan(const State &now_state, int fast = IS_FAST) {
	int dis = 0, now_num = 0, row = 0, col = 0;
	for (int i = 0; i < 16; ++i) {
		now_num = now_state.state[i];
		row = i / 4;
		col = i % 4;
		dis += abs(row - target_rows[now_num]) + abs(col - target_cols[now_num]);
		if (fast) {
			if (i < 15 && now_state.state[i + 1] != (now_state.state[i] + 1) % 16)dis += 4;
			if (now_state.state[i] == 0 && (row == 3 || col == 3) && !(row == 3 && col == 3)) {
				dis += 2;
			}
		} // the fast manhattan distance
	}
	return dis;
}

// calculate the Manhattan distance based on the last state
int calPerManhattan(const State &now_state, const State &next_state, int next_pos) {
	int a = now_state.state[now_state.zero_pos], b = now_state.state[next_pos];
	int now_row = now_state.zero_pos / 4, now_col = now_state.zero_pos % 4, next_row = next_pos / 4, next_col = next_pos % 4;
	int minus = abs(now_row - target_rows[a]) + abs(now_col - target_cols[a]) + abs(next_row - target_rows[b]) + abs(next_col - target_cols[b]),
		add = abs(next_row - target_rows[a]) + abs(next_col - target_cols[a]) + abs(now_row - target_rows[b]) + abs(now_col - target_cols[b]);
	return add - minus;
}

// judge if moving is valid.
bool isVaildMove(int pos, int i) {
	if (pos % 4 == 3 && i == 0)return false;
	if (pos > 11 && i == 1)return false;
	if (pos % 4 == 0 && i == 2)return false;
	if (pos < 4 && i == 3)return false;
	return true;
}

int road[250];
// the length of road
static int road_length = 0;
// direction
int moves[4] = { 1, 4, -1, -4 };

// dfs per time
int dfs(const State &now_state, int depth, int now_move_num, int max_depth, int last_state_h) {
	if (depth + last_state_h > max_depth) {
		return depth + last_state_h;
	}
	if (!last_state_h) {
		return -1;
	}
	int zero_pos = now_state.zero_pos, min_max_dis = INFINITY, move_sum = 0, next_pos;
	int next_move_num, max_dis, next_state_h;
	for (int i = 0; i < 4; ++i) {
		if (!isVaildMove(zero_pos, i))continue;
		next_pos = zero_pos + moves[i];
		next_move_num = now_state.state[next_pos];
		if (next_move_num == now_move_num)continue;

		State next_state(now_state);
		next_state.swap(zero_pos, next_pos);

		if (!IS_FAST)next_state_h = last_state_h + calPerManhattan(now_state, next_state, next_state.zero_pos);
		else next_state_h = calManhattan(next_state);

		max_dis = dfs(next_state, depth + 1, next_move_num, max_depth, next_state_h);
		road[road_length++] = next_move_num;
		if (max_dis == -1) {
			return -1;
		}
		min_max_dis = min(min_max_dis, max_dis);
		road_length--;
	}
	return min_max_dis;
}

// ida*
void idaStar(int state_list[]) {
	total_state.setState(state_list);
	int now_max_h = calManhattan(total_state), first_state_h = now_max_h;
	int step = 0;
	while (1) {
		now_max_h = dfs(total_state, 0, -1, now_max_h, first_state_h);
		/*printf("%d: %d\n", step, now_max_h);*/
		if (now_max_h == -1) {
			printf("\nRoad: \n");
			for (int i = 0; i < road_length; ++i) {
				printf("%d ", road[i]);
			}
			printf("\n");
			break;
		}
		road_length = 0;
		step += 1;
		total_state.setState(state_list);
	}
}

int main() {
	int begin_state[10][16] = { { 6, 1, 2, 4, 0, 7, 3, 8, 5, 9, 10, 11, 13, 14, 15, 12 },
	{ 14, 10, 6, 0, 4, 9, 1, 8, 2, 3, 5, 11, 12, 13, 7, 15 },
	{ 6, 10, 3, 15, 14, 8, 7, 11, 5, 1, 0, 2, 13, 12, 9, 4 },
	{ 11, 3, 1, 7, 4, 6, 8, 2, 15, 9, 10, 13, 14, 12, 5, 0 },
	{ 0, 5, 15, 14, 7, 9, 6, 13, 1, 2, 12, 10, 8, 11, 4, 3 },
	{10, 8, 9, 14, 0, 11, 15, 1, 5, 6, 3, 4, 7, 2, 12, 13},
	{2, 7, 9, 4, 15, 10, 3, 8, 12, 11, 5, 1, 6, 14, 0, 13},
	{15, 12, 2, 14, 8, 11, 9, 10, 5, 7, 13, 6, 1, 4, 0, 3},
	{11, 13, 3, 9, 1, 8, 5, 15, 12, 4, 6, 7, 10, 0, 14, 2},
	{8, 0, 5, 2, 3, 14, 9, 4, 6, 10, 1, 11, 7, 12, 13, 15} };
	double start = 0;
	IS_FAST = 0;// 1 => fast Manhattan
	for (int i = 0; i < 10; ++i) {
		printf("Test %d:", i);
		for (int j = 0; j < 16; ++j) {
			if (j % 4 == 0)printf("\n");
			printf("%d ", begin_state[i][j]);
		}
		start = clock();
		idaStar(begin_state[i]);
		printf("Total step:%d\n", road_length);
		printf("Time:%.5f seconds\n", (clock() - start) / CLOCKS_PER_SEC);
		printf("\n");
		road_length = 0;
	}
	system("pause");
	return 0;
}