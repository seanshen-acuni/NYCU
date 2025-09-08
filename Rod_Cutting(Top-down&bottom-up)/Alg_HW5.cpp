#include <iostream>
#include <random>
#include <ctime>
#include <vector>
const int Max =	 10000000;
const int Min =	-10000000;
using namespace std;

//top-down
struct resulta {
	int r1;
	vector<int> s1;
	int r2;
	vector<int> s2;
};
pair<int,int> Memorized_Cut_Rod_Aux(const vector<int>& p, int n, vector<int>& r1, vector<int>& s1, vector<int>& r2, vector<int>& s2) {
	
	if (r1[n] >= 0 && r2[n] >= 0) {
		return {r1[n],r2[n]};
	}
	int maxq = Min, minq = Max;
	if (n == 0) {
		maxq = 0;
		minq = 0;
	}
	else {
		for (int i = 1; i <= min(n, (int)p.size()); i++) {
			auto temp = Memorized_Cut_Rod_Aux(p, n - i, r1, s1, r2, s2);
			int max_temp = p[i - 1] + temp.first;
			int min_temp = p[i - 1] + temp.second;
			if (maxq < max_temp) {
				maxq = max_temp;
				s1[n] = i;
			}
			if (minq > min_temp) {
				minq = min_temp;
				s2[n] = i;
			}
//		s1[n] = smax;
//		s2[n] = smin;
		}
	}
	r1[n] = maxq;
	r2[n] = minq;
	return make_pair(maxq, minq);
}

resulta Memorized_Cut_Rod(const vector<int>& p, int n) {
	vector<int> r1(n + 1, Min);
	vector<int> r2(n + 1, Max);
	vector<int> s1(n + 1, 0);
	vector<int> s2(n + 1, 0);
	r1[0] = 0;
	r2[0] = 0;
	auto result = Memorized_Cut_Rod_Aux(p, n, r1, s1, r2, s2);
	return { result.first, s1, result.second, s2 };
}

void Print_Cut_Rod_Solution_Top_Down(const vector<int>& p, int n) {
	auto result = Memorized_Cut_Rod(p, n);
	int rmax = result.r1;
	int rmin = result.r2;
	int n1 = n;
	vector<int> smax = result.s1;
	vector<int> smin = result.s2;
	cout << "Maximum Revenue:" << rmax << endl;
	cout << "Cutting Positions(max):";
	int imax=0;
	while (n > 0){
		cout << smax[n] << " ";
		n -= smax[n];
		imax++;
	}
	cout << endl << "cut into "<< imax <<" parts" << endl;
	cout << "Minimum Revenue:" << rmin << endl;
	cout << "Cutting Positions(min):";
	int imin = 0;
	n = n1;
	while (n > 0) {
		cout << smin[n] << " ";
		n -= smin[n];
		imin++;
	}
	cout << endl << "cut into " << imin << " parts" << endl;
}
//bottom-up
struct resultb {
	vector<int> r1;
	vector<int> s1;
	vector<int> r2;
	vector<int> s2;
};

resultb Extended_Bottom_Up_Cut_Rod(vector<int>& p, int n) {
//	vector<int> r(n + 1	, 0);
//	vector<int> s(n + 1 , 0);
	vector<int> r1(n + 1, 0);
	vector<int> r2(n + 1, 0);
	vector<int> s1(n + 1, 0);
	vector<int> s2(n + 1, 0);
	int max_q = 0;
	int min_q = 0;
//	int q = 0;
	for (int j = 1; j <= n; j++) {
		//initialize q everytime
		max_q = Min;
		min_q = Max;
		for (int i = 1; i <= min(j, (int)p.size()); i++) {
			if (max_q < p[i-1] + r1[j - i]) {
				max_q = p[i-1] + r1[j - i];
				s1[j] = i;
			}
			if (min_q > p[i - 1] + r2[j - i]) {
				min_q = p[i - 1] + r2[j - i];
				s2[j] = i;
			}
		}
		r1[j] = max_q;
		r2[j] = min_q;
	}
	return { r1, s1, r2 ,s2 };
}
void Print_Cut_Rod_Solution_Bottom_up(vector<int>& p, int n) {
	auto result = Extended_Bottom_Up_Cut_Rod(p, n);
	vector<int> rmax = result.r1;
	vector<int> rmin = result.r2;
	vector<int> smax = result.s1;
	vector<int> smin = result.s2;
	int imax = 0, imin = 0, n1 = n;
	cout << "Maximum Revenue:" << rmax[n] << endl;
	cout << "Cutting Positions(max):";
	while (n > 0) {
		cout << smax[n] << " ";
		n -= smax[n];
		imax++;
	}
	cout << endl << "cut into " << imax << " parts" << endl;
	n = n1;
	//intialize n to original value
	cout << "Minimum Revenue:" << rmin[n] << endl;
	cout << "Cutting Positions(min):";
	while (n > 0) {
		cout << smin[n] << " ";
		n -= smin[n];
		imin++;
	}
	cout << endl << "cut into " << imin << " parts" << endl;
}
int main(){
	int n;
	double time1, time2, time3, time4;
	cout << "Enter the length of the rod:";
	cin >> n;
	cout << "Please enter the prices for each length(1~10):" << endl;
	vector<int> p(10);
	for (int i = 0; i < 10; i++) {
		cin >> p[i];
	}
	cout << "total length: " << n << endl << endl;
	cout << "Top-Down results:" << endl;
	time1 = clock();
	Print_Cut_Rod_Solution_Top_Down(p, n);
	time2 = clock();
	cout << "compile time(top-down): " << time2 - time1 << endl << endl;
	cout << "Bottom-up results:" << endl;
	time3 = clock();
	Print_Cut_Rod_Solution_Bottom_up(p, n);
	time4 = clock();
	cout << "compile time(bottom-up): " << time4 - time3;
	return 0;
}