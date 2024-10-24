#include <iostream>
#include <ostream>
#include <vector>
#include <chrono>  // Per misurare il tempo
typedef long long int ll;
using namespace std;

// Funzione per unire due sotto-array ordinati
void merge(vector<ll> &arr, int left, int mid, int right) {
    int n1 = mid - left + 1;  // Lunghezza del primo sotto-array
    int n2 = right - mid;     // Lunghezza del secondo sotto-array

    // Usa std::vector invece di array allocati nello stack
    vector<ll> L(n1), R(n2);

    // Copia i dati negli array temporanei L[] e R[]
    for (int i = 0; i < n1; i++)
        L[i] = arr[left + i];
    for (int j = 0; j < n2; j++)
        R[j] = arr[mid + 1 + j];

    // Indici per iterare su L[], R[] e arr[]
    int i = 0, j = 0, k = left;

    // Unisci i due sotto-array ordinati
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    // Copia gli elementi rimanenti di L[], se ci sono
    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    // Copia gli elementi rimanenti di R[], se ci sono
    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }
}

// Funzione Merge Sort parallelizzata
void mergeSort(vector<ll> &arr, int left, int right, int depth) {
    if (left < right) {
        int mid = left + (right - left) / 2;

        // Parallelizziamo solo se il sotto-array è sufficientemente grande
        // e la profondità della ricorsione non è eccessiva
        if (depth < 4) {
            #pragma omp task shared(arr)
            mergeSort(arr, left, mid, depth + 1);

            #pragma omp task shared(arr)
            mergeSort(arr, mid + 1, right, depth + 1);

            #pragma omp taskwait
            merge(arr, left, mid, right);

        } else {
            mergeSort(arr, left, mid, depth + 1);
            mergeSort(arr, mid + 1, right, depth + 1);
            merge(arr, left, mid, right);
        }
    }
}

// Funzione di Merge Sort non parallela
void mergeSortSequential(vector<ll> &arr, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;

        mergeSortSequential(arr, left, mid);
        mergeSortSequential(arr, mid + 1, right);

        merge(arr, left, mid, right);
    }
}

// Funzione per stampare un array
void printArray(const vector<int> &arr) {
    for (int elem : arr)
        cout << elem << " ";
    cout << endl;
}

// assegno un vector globale
vector<ll> generateRandomArray(ll size) {
    vector<ll> arr(size);
    for (int i = 0; i < size; i++) {
        arr[i] = rand() % size;
    }
    return arr;
}

int main() {
    ll n;
    cout << "size del array: ";
    cin >> n;

    vector<ll> arr = generateRandomArray(n);

    // Inizio della misurazione del tempo
    auto start = chrono::high_resolution_clock::now();

    #pragma omp parallel
    {
        #pragma omp single
        mergeSort(arr, 0, arr.size() - 1, 0);  // La profondità inizia da 0
    }

    // Fine della misurazione del tempo
    auto end = chrono::high_resolution_clock::now();

    // Calcolo del tempo totale
    chrono::duration<double> duration = end - start;
    cout << "Tempo di esecuzione(parallela): " << duration.count() << " secondi" << endl;

    arr = generateRandomArray(n); // ri-randomizzo gli elementi

    // ---------------------- Merge Sort Sequenziale ----------------------
    auto start_seq = chrono::high_resolution_clock::now();
    mergeSortSequential(arr, 0, arr.size() - 1);
    auto end_seq = chrono::high_resolution_clock::now();
    chrono::duration<double> duration_seq = end_seq - start_seq;
    cout << "Tempo di esecuzione (sequenziale): " << duration_seq.count() << " secondi" << endl;

    return 0;
}