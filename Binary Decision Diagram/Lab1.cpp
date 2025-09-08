#include <iostream>
#include <fstream>
#include <unordered_map>
#include <vector>
#include <string>
#include <set>
#include <climits>
#include <algorithm>
#include <sstream>

using namespace std;

struct BDDNode {
    int var;
    BDDNode *low, *high;
    //new BDDNode(var, low, high)
    BDDNode(int v, BDDNode* l, BDDNode* h) : var(v), low(l), high(h) {}
};

// Hash table for unique nodes
unordered_map<string, BDDNode*> nodeTable;

// Memoization for reducing nodes
BDDNode* getNode(int var, BDDNode* low, BDDNode* high) {
    if (var >= 0 && low == high) return low; // Rule1: If both children are the same, return one
    string key = to_string(var) + "-" + to_string(reinterpret_cast<uintptr_t>(low)) + "-" + to_string(reinterpret_cast<uintptr_t>(high));
    if (nodeTable.find(key) != nodeTable.end())
        return nodeTable[key];

    BDDNode* newNode = new BDDNode(var, low, high);
    nodeTable[key] = newNode;
    return newNode;
}


bool evaluateExpr(const string& expr, const unordered_map<char, bool>& assignment) {
    // Handle constant expressions
    if (expr == "1") return true;
    if (expr == "0") return false;
    
    bool result = false;
    size_t start = 0;
    
    while (start < expr.size()) {
        // OR
        size_t plusPos = expr.find('+', start);
        string term;
        if (plusPos == string::npos) {
            term = expr.substr(start);
            start = expr.size();
        } else {
            term = expr.substr(start, plusPos - start);
            start = plusPos + 1;
        }
        
        // AND
        bool termResult = true;
        for (char c : term) {
            bool transferlow;
            if(isupper(c))
                transferlow = !assignment.at(tolower(c));
            else
                transferlow = assignment.at(c);
        termResult = termResult && transferlow;
        }
        result = result || termResult;
    }
    return result;
}

// Recursive BDD construction
BDDNode* buildBDD(const string& func, int varIndex, const vector<char>& order, unordered_map<char, bool>& assignment){
    if (varIndex >= order.size()) {
        bool value = evaluateExpr(func, assignment);
        return value ? getNode(-1, nullptr, nullptr) : getNode(-2, nullptr, nullptr);
    }

    char var = order[varIndex];
    // Low branch: assign false to the current variable.
    assignment[var] = false;
    BDDNode* low = buildBDD(func, varIndex + 1, order, assignment);
    
    // High branch: assign true to the current variable.
    assignment[var] = true;
    BDDNode* high = buildBDD(func, varIndex + 1, order, assignment);

    return getNode(var, low, high);
}
// || root->var < 0
// Count unique nodes
void countNodes(BDDNode* root, set<BDDNode*>& uniqueNodes) {
    if (!root) return;
    if (uniqueNodes.find(root) != uniqueNodes.end()) return;
    
    uniqueNodes.insert(root);
    countNodes(root->low, uniqueNodes);
    countNodes(root->high, uniqueNodes);
}

int main(int argc, char* argv[]){
    //data reading
    if(argc < 3){
        cerr << "Usage: ./Lab1 input.txt output.txt" << endl;
        return 1;
    }
    ifstream inputFile(argv[1]);
    ofstream outputFile(argv[2]);

    if (!inputFile || !outputFile) {
        cerr << "Error opening files!" << endl;
        return 1;
    }
    string fileContent((istreambuf_iterator<char>(inputFile)), istreambuf_iterator<char>());
    fileContent.erase(remove(fileContent.begin(), fileContent.end(), '\n'), fileContent.end());
    stringstream ss(fileContent);
    string func;
    getline(ss, func, '.');

    vector<vector<char>> orderings;
    string order;
    while(getline(ss, order, '.')){
        orderings.push_back(vector<char>(order.begin(),order.end()));
    }
// ordering:{'a','c','b','d'}
    int minNodes = INT_MAX;
    for(const auto& ordering : orderings){
        nodeTable.clear();
        unordered_map<char, bool> assignment;
        BDDNode* root = buildBDD(func, 0, ordering,assignment);
        set<BDDNode*> uniqueNodes;
        countNodes(root, uniqueNodes);
        minNodes = min(minNodes, (int)uniqueNodes.size());
    }

    outputFile << minNodes << endl;
    inputFile.close();
    outputFile.close();
    return 0;
}