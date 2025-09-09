#include <iostream>
#include <fstream>
#include <queue>
using namespace std;
const int Max = 10000;
const int n = 17;
const int dx[] = {1, 0, -1, 0};  // right, down, left, up
const int dy[] = {0, 1, 0, -1};
const int m = n * n;

ofstream outputFile("output.txt");

struct Vertex {
    int color; // 0: white, 1: gray, 2: black
    int d;
    int pi;
};

vector<Vertex> vertices(m);

void print_path(int s, int v){
	if(v == s){
		outputFile << "{"<< s / n + 1 << "," << s % n + 1 << "}" << endl;
	}else if (vertices[v].pi == -1){
		outputFile << "no path from " << s << " to " << v << "exists!";
	}else{
		print_path(s, vertices[v].pi);
		outputFile << "{"<< v / n + 1 << "," << v % n + 1 << "}" << endl;
	}
}
void BFS(const int G[n][n], int s) {//s:0~17^2-1(position)
    // Initialize all vertices except the source
    for (int i = 0; i < m; i++) {
        if (i != s) {
            vertices[i].color = 0; // white
            vertices[i].d = Max; // infinity
            vertices[i].pi = -1; // NIL
        }
    }
    // Initialize the source vertex
    vertices[s].color = 1; // gray
    vertices[s].d = 0;
    vertices[s].pi = -1; // NIL
    queue<int> Q;
    Q.push(s);
    
    while (!Q.empty()) {
        int u = Q.front();
        Q.pop();
//      outputFile << u << " "; 
		int ux = u / n;
		int uy = u % n;
        // Explore all adjacent vertices of u
        for (int i = 0; i < 4; i++) {
        	int newx = ux + dx[i];
            int newy = uy + dy[i];
            int v = newx * n + newy;
            if (newx >= 0 && newx < n && newy >= 0 && newy < n && vertices[v].color == 0) {
        		if (G[newx][newy] == 2){
        			vertices[v].color = 1; // gray
        			vertices[v].d = vertices[u].d + 2; 
        			vertices[v].pi = u;
            		Q.push(v);
				}else if (G[newx][newy] == 1) { // if v is white
           	    	vertices[v].color = 1; // gray
            		vertices[v].d = vertices[u].d + 1;
             		vertices[v].pi = u;
            		Q.push(v);
           		}
        	}
        	
        }
        
		vertices[u].color = 2; // black
    }
	outputFile << "step=" << vertices[m - 1].d << endl;
	print_path(s, m - 1);
}



int main() {
    ifstream inputFile("input.txt");
    if (!inputFile.is_open()) {
        cerr << "Error opening file!" << endl;
        return 1;
    }
    
    for (int t = 0; t < 20; t++) {
    	int maze[n][n];
        // Read the maze from input file
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                inputFile >> maze[i][j];
//                outputFile << maze[i][j]; 
            }
//            outputFile << endl;
        }
        int s = 0;
        outputFile << "pattern " << t + 1 << endl;
		BFS(maze, s);
		outputFile << endl;
    }

    inputFile.close();
    outputFile.close();

    return 0;
}

