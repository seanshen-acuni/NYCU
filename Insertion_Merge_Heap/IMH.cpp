#include <iostream>
#include <random>
#include <ctime>
#include <vector>
const int Max = 100000000;
using namespace std;
//merge sort
void merge(vector<int>& array, int l, int m, int r) {

    int const nr = m - l + 1;
    int const nl = r - m;

    //put array[l]~array[m] into leftsubarray
    vector<int> leftsubarray(array.begin() + l, array.begin() + m + 1);
    //put array[m+1]~array[r] into rightsubarray
    vector<int> rightsubarray(array.begin() + m + 1, array.begin() + r + 1);
    //because vectors are left-closed,right-opened

    //insert maximum into the end of leftsubarray
    leftsubarray.insert(leftsubarray.end(), Max);
    //insert maximum into the end of rightsubarray
    rightsubarray.insert(rightsubarray.end(), Max);

    //other method(this method need to for(){}*2 before insert/pushback maximum
    /*
        for (int i = 1; i <= nr; i++) {
            leftsubarray[i] = array[l + i - 1];
        }
        for (int j = 1; j <= nl; j++) {
            rightsubarray[j] = array[m + j];
        }
    */
    ///*
    int i = 0, j = 0;
    for (int k = l; k <= r; k++) {
        //left smaller than right -> left first
        if (leftsubarray[i] <= rightsubarray[j]) {
            array[k] = leftsubarray[i];
            i++;
        }
        //right smaller than left -> right first
        else {
            array[k] = rightsubarray[j];
            j++;
        }
    }
    //*/
}
void merge_sort(vector<int>& array, int l, int r) {
    if (l < r) {
        int m = (l + r) / 2;
        merge_sort(array, l, m);
        merge_sort(array, m + 1, r);
        merge(array, l, m, r);
    }
}
//head sort
void Max_Heapify(vector<int>& A, int i,int heap_size) {

    int l = 2 * i + 1; // Left child index
    int r = 2 * i + 2; // Right child index
    int largest = i;   // Initialize largest as root

    // Compare left child with root
    if (l < heap_size && A[l] > A[largest])
        largest = l;

    // Compare right child with largest so far
    if (r < heap_size && A[r] > A[largest])
        largest = r;

    // If largest is not root, swap and continue heapifying
    if (largest != i) {
        swap(A[i], A[largest]);
        Max_Heapify(A, largest, heap_size);
    }
}

void Build_Max_Heap(vector<int>& A,int heap_size) {

    // Build max heap by calling MaxHeapify on non-leaf nodes
    for (int i = heap_size / 2 - 1 ; i >= 0; i--) {
        Max_Heapify(A, i, heap_size);
    }
}

void Heapsort(vector<int>& A) {
    //build a max heap first
    int heap_size = A.size();
    Build_Max_Heap(A, heap_size);
    for (int i = heap_size - 1 ; i >= 0; i--) {
        //swap the last element with the root of the heap
        swap(A[0], A[i]);
        heap_size--;
        //the root of heap is not maximun, need to compile maxheapify
        Max_Heapify(A, 0, heap_size);
    }
}

void random_vector_generator(vector<int>& array) {
    //fill the array with random numbers
    for (int i = 0; i < array.size(); i++) {
        array[i] = rand();
        //array.push_back(std::rand());
        //it can work even if we doesn't declare the size of array in cpp 11
    }
}
int main() {
    srand(time(0));
    //set input "size"
    //set array as a vector to let the size of array be definable by "size"
    int size = 0;
    cout << "Enter the size of the array: ";
    cin >> size;
    vector<int>array(size);
    random_vector_generator(array);
    vector<int>arrayb = array;
    vector<int>arrayc = array;
    /*
        vector<int>array(size);
        //fill the array with random numbers
        for (int i = 0; i < size; i++) {
            array[i] = rand();
            //array.push_back(std::rand());
            //it can work even if we doesn't declare the size of array in cpp 11
        }
        vector<int>arrayb = array;
    */
    //array generating test
    /*
    //Print the array
    std::cout << "Generated array:";
    for (int i = 0; i < size; i++) {
        std::cout << array[i] << " ";
    }
    return 0;
    */

    ///*
       //insertion sort
    double START1, END1, START2, END2, START3, END3;
    START1 = clock();
    int key, k;
    for (int j = 1; j < size; j++) {
        key = array[j];
        k = j - 1;
        while ((k >= 0) && (array[k] > key)) {
            array[k + 1] = array[k];
            k = k - 1;
        }
        array[k + 1] = key;
    }
    END1 = clock();
/*
        cout << "Generated array(by insertion sort):" << endl;
        for (int i = 0; i < size; i++) {
            cout << array[i] << " ";
        }
        cout << endl;
*/    
    cout << "insertion-sorting run time:" << (END1 - START1) << endl;


        //merge sort
    START2 = clock();
    merge_sort(arrayb, 0, size - 1);
    END2 = clock();
   /*
        cout << "Generated array(by merge sort):" << endl;
        for (int i = 0; i < size; i++) {
            cout << arrayb[i] << " ";
        }
        cout << endl;
    */
    //head sort
    cout << "merge-sorting run time:" << (END2 - START2) << endl;
    START3 = clock();
    Heapsort(arrayc);
    END3 = clock();
    cout << "heapsorting run time:" << (END3 - START3) << endl;
    
    /*        cout << "Generated array(by heapsort):" << endl;
        for (int i = 0; i < size; i++) {
            cout << arrayc[i] << " ";
        }
        cout << endl;
    */
    return 0;
    
}
