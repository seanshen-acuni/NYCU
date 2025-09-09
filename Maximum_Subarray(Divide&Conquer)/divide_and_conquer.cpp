#include <iostream>
#include <random>
#include <ctime>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <tuple>
#include <set>
const int Max = 100000000;
using namespace std;
set<pair<int, int>>same;
float current_max = -Max;

tuple<int,int,float> Find_maximum_cross_subarray(vector<float> &A, int l , int m, int h) {
    float left_sum = -Max;
    float sum1 = 0, sum2 = 0;
    int max_left = l, max_right = h;
    for (int i = m; i >= l; i--) {
        sum1 += A[i];
        if (sum1 > left_sum) {
            left_sum = sum1;
             max_left = i;
        }
    }
    float right_sum = -Max;
    for (int j = m + 1; j <= h; j++){
        sum2 += A[j];
        if (sum2 > right_sum) {
            right_sum = sum2;
            max_right = j;
        }
    }
    float ans = right_sum + left_sum;
    return{ max_left, max_right, ans };
}

void filter(float maximum,int l,int h) {
    if (maximum > current_max) {
        same.clear();
        same.insert(make_pair(l, h));
        
    }
    else if (maximum == current_max) {
        same.insert(make_pair(l, h));
    }
    current_max = maximum;
}

tuple<int,int,float> Find_maximum_subarray(vector<float> &A,int l, int h) {
    if (h == l) {
        return { l,h,A[l] };
    }
    int m = (l + h) / 2;
    auto left_result = Find_maximum_subarray(A, l, m);
    auto right_result = Find_maximum_subarray(A, m + 1, h);
    auto cross_result = Find_maximum_cross_subarray(A, l, m, h);

    int left_low, left_high, right_low, right_high, cross_low, cross_high;
    float left_sum, right_sum, cross_sum;
    //store the result in
    tie(left_low, left_high, left_sum) = left_result;
    tie(right_low, right_high, right_sum) = right_result;
    tie(cross_low, cross_high, cross_sum) = cross_result;
    
    if (left_sum >= right_sum && left_sum >= cross_sum) {
        filter(left_sum, left_low, left_high);
        return{ left_low, left_high, left_sum };
    }
    else if (right_sum >= left_sum && right_sum >= cross_sum) {
        filter(right_sum, right_low, right_high);
        return{ right_low, right_high, right_sum };
    }
    else if (cross_sum >= right_sum && cross_sum >= left_sum) {
        filter(cross_sum, cross_low, cross_high);
        return{ cross_low, cross_high, cross_sum };
    } 
}

int main()
{
    //read the file "data.txt/data2.txt"
    ifstream infile("data2.txt");
    if (!infile) {
        cout << "Failed to open file!!" << endl;
        return 1;
    }
    //store the data into string data(one line in one time)
    string data;
    vector<float> array;
    int cnt = 0;
    while (getline(infile, data)) {
        //cnt++;
        //there's two float numbers in every line -> num1,num2
        //if (cnt < 110000) continue;
        istringstream iss(data);
        float num1, num2;
        if (iss >> num1 >> num2) {
            //store only num2 into array, num1 is index
            array.push_back(num2);
            //cout << num2 << endl;
        }
        
    }
    infile.close();
    auto max_result = Find_maximum_subarray(array, 0, array.size() - 1);
    int max_low, max_high;
    float max_sum;
    tie(max_low, max_high, max_sum) = max_result;
    for (auto p : same)
    {
        cout << "start from " << p.first+1 << "to " << p.second+1 << endl;
    }
    //cout << "start from " << max_low+1 << "to " << max_high+1 << endl;
    //cout << "maximum sum is " << max_sum;
    cout << "maximum sum is " << current_max;
    return 0;


}
