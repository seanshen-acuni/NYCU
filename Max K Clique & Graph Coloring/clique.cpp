#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;
#define fastio ios::sync_with_stdio(false), cin.tie(NULL), cout.tie(NULL)
#pragma GCC optimize("Ofast,unroll-loops,no-stack-protector,fast-math")
#pragma GCC target("fma,sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,avx,avx2,tune=native")
#pragma comment(linker, "/stack:200000000")

struct input {
	vector<int> a;
	vector<int> b;
};

input ip;
vector<vector<int>> record_list, path_list;
vector<int> record, num_list, temp;
bool choosed[1000] = { false }, zero = false;

void backtracking(int start, int level) {

	//cout <<"level:" << level << " start:" << start << endl;

	bool end = false;
	int road = 0;

	if (start != 0) {
		for (int num : path_list[start]) {

			int n_same = 0;

			if (count(record.begin(), record.end(), num) != 0) continue;

			road++;

			for (int num2 : path_list[num]) {

				for (int num3 : record) {

					if (num3 == num2)
						n_same++;
				}
			}
			//cout <<"num:" << num << " n_same:" << n_same << endl;
			if (n_same != record.size()) {
				road--;
			}
		}
		if (road == 0)
			end = true;
	}
	//if (record.size() == 3)
	//	end = true;
	if (end) {
		//cout <<"out" << endl;
		//if (record.size() <= 2)
		//	return;
		//temp.clear();
		//temp.reserve(record.size());
		//temp.insert(temp.end(), record.begin(), record.end());
		//sort(temp.begin() , temp.end());
		//cout << "output:";
		//for (int num : record)
		//	cout << num <<" ";
		//cout << endl;

		record_list.emplace_back(record);

		return;
	}
	//cout <<"deep level:" << level << " start:" << start << endl;

	int index = 0;

	for (path_list[start][index]; index < path_list[start].size(); index++) {
		bool dis = false;
		int same = 0;

		//cout << "temp:" << path_list[start][index] << endl;

		if (count(record.begin(), record.end(), path_list[start][index]) != 0 || path_list[start][index] < start) continue;

		//cout << "temp in:" << path_list[start][index] << endl;

		record.emplace_back(path_list[start][index]);

		if (record.size() >= 1) {
			for (int num : path_list[path_list[start][index]]) {
				for (int num2 : record) {
					//cout << "start:" << start << "next:" << path_list[start][index] << "num:" << num << "record" << num2 << endl;
					if (num == num2)
						same++;
				}
			}
			//cout << record.size() <<"same:" << same << endl;
			if (same != record.size() - 1 && record.size() > 1) {
				record.pop_back();
				dis = true;
			}
		}
		if (dis) continue;
		/*
		for (int num : record)
			cout << num <<" ";
		cout << endl;
		*/
		backtracking(path_list[start][index], level + 1);
		choosed[path_list[start][index] - 1] = false;

		record.pop_back();
	}
}

int main() {

	fastio;

	int x, y;

	while (cin >> x >> y) {

		if (x == 0 || y == 0)
			zero = true;

		ip.a.emplace_back(x);
		ip.b.emplace_back(y);
	}

	if (zero) {

		for (int i = 0; i < ip.a.size(); i++) {
			ip.a[i] ++;
			ip.b[i] ++;
		}
	}

	bool ip_num[10000] = { false };

	for (auto num : ip.a) {
		if (ip_num[num - 1] == false) {
			ip_num[num - 1] = true;
			num_list.emplace_back(num);
		}
	}
	for (auto num : ip.b) {
		if (ip_num[num - 1] == false) {
			ip_num[num - 1] = true;
			num_list.emplace_back(num);
		}
	}
	path_list.emplace_back(num_list);

	for (int start : num_list) {

		//temp.emplace_back(start);

		for (int i = 0; i < ip.b.size(); i++) {
			if (ip.a[i] == start)
				if (count(temp.begin(), temp.end(), ip.b[i]) == 0)
					temp.emplace_back(ip.b[i]);
			if (ip.b[i] == start) {
				if (count(temp.begin(), temp.end(), ip.a[i]) == 0)
					temp.emplace_back(ip.a[i]);
			}
		}
		path_list.emplace_back(temp);
		temp.clear();
	}


	/*
	for (int i = 0 ; i < path_list.size() ; i ++) {
		for (int j = 0 ; j < path_list[i].size() ; j ++) {
			cout << path_list[i][j] << " ";
		}
		cout << endl;
	}
	cout << "======================================" << endl;
	*/
	backtracking(0, 0);

	if (zero) {
		for (int i = 0; i < record_list.size(); i++) {
			for (int j = 0; j < record_list[i].size(); j++) {
				record_list[i][j] --;
			}
		}
	}
	int max_clique = 0;
	for (int i = 0; i < record_list.size(); i++) {
		if (record_list[i].size() > max_clique)
			max_clique = record_list[i].size();
	}

	for (int t = 3; t <= max_clique; t++) {
		cout << t << endl;
		for (int i = 0; i < record_list.size(); i++) {
			if (record_list[i].size() == t) {
				cout << "{";
				for (int j = 0; j < record_list[i].size(); j++) {
					cout << record_list[i][j];
					if (j != record_list[i].size() - 1)
						cout << ",";
				}
				cout << "}" << endl;
			}
		}
	}


	return 0;
}
