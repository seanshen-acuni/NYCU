#include <iostream>                                                                                                                                xC                                                                                                          XXXXCX                                    
#include <random>
#include <ctime>
#include <vector>
using namespace std;

pair<int,vector<vector<vector<int>>>> LCS_Length(const vector<float>& X, const vector<float>& Y, const vector<float>& Z) {
	int m = X.size();
	int n = Y.size();
	int o = Z.size();
	//X:[0~m-1],Y:[0~n-1],Z:[0~o-1]
	// Initialize 3D vector for directions
	vector<vector<vector<int>>> b(m + 1, vector<vector<int>>(n + 1, vector<int>(o + 1, 0)));
	// Initialize 3D vector for LCS length calculations
	vector<vector<vector<int>>> C(m + 1, vector<vector<int>>(n + 1, vector<int>(o + 1, 0)));
	//fill C,b
	for (int i = 1; i <= m; i++){
		for (int j = 1; j <= n; j++){
        	for (int k = 1; k <= o; k++){
            	if((X[i - 1] == Y[j - 1]) && (X[i - 1] == Z[k - 1])){
            		C[i][j][k] = C[i - 1][j - 1][k - 1] + 1;
            		b[i][j][k] = 5;//v3
				}
				/*
				else if(C[i - 1][j][k] >= C[i][j][k - 1] && C[i][j - 1][k] >= C[i][j][k - 1] && X[i - 1] == Y[j - 1]){
					C[i][j][k] = C[i - 1][j - 1][k];
					b[i][j][k] = 2;
				}
				else if(C[i - 1][j][k] >= C[i][j - 1][k] && C[i - 1][j][k] >= C[i][j][k - 1] && Y[j - 1] == Z[k - 1]){
					C[i][j][k] = C[i][j - 1][k - 1];
					b[i][j][k] = 4;
				}
				else if(C[i - 1][j][k] >= C[i][j - 1][k] && C[i][j][k - 1] >= C[i][j - 1][k] && X[i - 1] == Z[k - 1]){
					C[i][j][k] = C[i - 1][j][k - 1];
					b[i][j][k] = 6;
				}
				*/
				else if(C[i][j - 1][k] >= C[i - 1][j][k] && C[i][j - 1][k] >= C[i][j][k - 1]){
					C[i][j][k] = C[i][j-1][k];
					b[i][j][k] = 1;//forward Y
				}
				else if(C[i - 1][j][k] >= C[i][j - 1][k] && C[i - 1][j][k] >= C[i][j][k - 1]){
					C[i][j][k] = C[i - 1][j][k];
					b[i][j][k] = 3;//forward X
				}
				else if(C[i][j][k - 1] >= C[i - 1][j][k] && C[i][j][k - 1] >= C[i][j - 1][k]){
					C[i][j][k] = C[i][j][k - 1];
					b[i][j][k] = 7;//forward Z
				}
			}
		}
	}
	return make_pair(C[m][n][o],b); 
}

void Print_LCS(const vector<vector<vector<int>>>& b, vector<float> &X, int i, int j, int k) {
    if (i == 0 || j == 0 || k == 0) {
        return;
    }
    if (b[i][j][k] == 5) {
        Print_LCS(b, X, i - 1, j - 1, k - 1);
        cout << X[i - 1] << " ";
    } else if (b[i][j][k] == 3) {
        Print_LCS(b, X, i - 1, j, k);
    } else if (b[i][j][k] == 1) {
        Print_LCS(b, X, i, j - 1, k);
    } else {
        Print_LCS(b, X, i, j, k - 1);
    }
}       
            
int main() {
    vector<float> X,Y,Z;
	float temp;
	//input vector X
	cout << "X=?" << endl;
	while(cin >> temp){
		if(cin.get() == '\n')
			break;
		X.push_back(temp);
	}
	X.push_back(temp);
	//input vector Y
    cout << "Y=?" << endl;
    while(cin >> temp){
		if(cin.get() == '\n')
			break;
		Y.push_back(temp);
	}
	Y.push_back(temp);
	//input vector Z
    cout << "Z=?" << endl;
    while(cin >> temp){
		if(cin.get() == '\n')
			break;
		Z.push_back(temp);
	}
	Z.push_back(temp);
	//calculate LCS
    auto 	result = LCS_Length(X, Y, Z);
    int 	length = result.first;
	vector<vector<vector<int>>> b = result.second;
    cout << "LCS: ";
    Print_LCS(b, X, X.size(), Y.size(), Z.size());
    cout << endl;
	cout << "LCS Length: " << length << endl;
    return 0;
}

