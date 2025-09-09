#include <iostream>
#include <random>
#include <ctime>
#include <vector>
#include <iomanip> 
using namespace std;
const int Max =	 100;
pair<vector<vector<float>>,vector<vector<int>>> Optimal_BST(const vector<float> &p,const vector<float> &q,int n){
	vector<vector<float>> 	e(n + 2, vector<float>(n + 1, 0));
	vector<vector<float>> 	w(n + 2, vector<float>(n + 1, 0));
	vector<vector<int>> 	root(n + 1, vector<int>(n + 1, 0));
	for (int i = 1; i <= n + 1; i++){
		e[i][i - 1] = q[i - 1];
		w[i][i - 1] = q[i - 1];
	}
	for (int l = 1; l <= n; l++){
		for(int i = 1; i <= n - l + 1; i++){
			int j = i + l - 1;
			e[i][j] = Max;
			w[i][j] = w[i][j - 1] + p[j - 1] + q[j];
			
			for(int r = i; r <= j; r++){
				float t = e[i][r - 1] + e[r + 1][j] + w[i][j];
				
				if(t < e[i][j]){
					e[i][j]		= t;
					root[i][j]	= r;
				}
			}
		}
	}
	return make_pair(e, root);
}
void Print_Optimal_BST(const vector<vector<int>> &root, int i, int j, int depth = 0) {
    if (i <= j) {
    	//son valid 
        cout << string(depth * 2, ' ') << "k" << root[i][j] << endl;
        Print_Optimal_BST(root, i, root[i][j] - 1, depth + 1);
        Print_Optimal_BST(root, root[i][j] + 1, j, depth + 1);
    } else {
        cout << string(depth * 2, ' ') << "d" << j << endl;
    }
}
int main() {
    vector<float> p = {0.05, 0.04, 0.02, 0.07, 0.08, 0.09, 0.04, 0.08, 0.03};
    vector<float> q = {0.08, 0.06, 0.04, 0.04, 0.03, 0.06, 0.07, 0.06, 0.04, 0.02};
	int n = p.size();
    auto result = Optimal_BST(p, q, n);
    vector<vector<float>> e = result.first;
    vector<vector<int>>	root = result.second;
    cout << "p = ";
    for(int i = 0; i <= n - 1; i++){
    	cout << fixed << setprecision(2) << p[i] << " ";
	}
	cout << endl << "q = ";
	for(int i = 0; i <= n; i++){
		cout << q[i] << " ";
	}
	cout << endl;
    cout << "Smallest search cost: " << e[1][n] << endl;
    cout << "Root: " << root[1][n] << endl;
    Print_Optimal_BST(root, 1, n);
	return 0;
    
}
