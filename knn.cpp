#include <bits/stdc++.h>
#include <omp.h>
using namespace std;
using namespace std::chrono;

// Generate random 2D training data (x, y) with binary labels (0 or 1)
void generate_data(vector<vector<double>>& features, vector<int>& labels, int n) {
    srand(time(0));
    for (int i = 0; i < n; ++i) {
        double x = rand() % 1000;
        double y = rand() % 1000;
        features.push_back({x, y});
        labels.push_back((x + y > 1000) ? 1 : 0);  // Rough separation
    }
}

// Euclidean distance between two points
double euclidean_distance(const vector<double>& a, const vector<double>& b) {
    return sqrt((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]));
}

// Predict a single point (Sequential)
int predict_single(const vector<vector<double>>& train_X, const vector<int>& train_Y, const vector<double>& test_point, int k) {
    vector<pair<double, int>> distances;
    for (int i = 0; i < train_X.size(); ++i) {
        double dist = euclidean_distance(train_X[i], test_point);
        distances.push_back({dist, train_Y[i]});
    }
    sort(distances.begin(), distances.end());
    int count0 = 0, count1 = 0;
    for (int i = 0; i < k; ++i) {
        if (distances[i].second == 0) count0++;
        else count1++;
    }
    return (count1 > count0) ? 1 : 0;
}

// Sequential KNN prediction
void knn_sequential(const vector<vector<double>>& train_X, const vector<int>& train_Y,
                    const vector<vector<double>>& test_X, vector<int>& predicted, int k) {
    for (int i = 0; i < test_X.size(); ++i) {
        predicted[i] = predict_single(train_X, train_Y, test_X[i], k);
    }
}

// Parallel KNN prediction
void knn_parallel(const vector<vector<double>>& train_X, const vector<int>& train_Y,
                  const vector<vector<double>>& test_X, vector<int>& predicted, int k) {
    #pragma omp parallel for
    for (int i = 0; i < test_X.size(); ++i) {
        predicted[i] = predict_single(train_X, train_Y, test_X[i], k);
    }
}

int main() {
    int train_size = 10000, test_size = 1000, k = 5;

    vector<vector<double>> train_X, test_X;
    vector<int> train_Y, test_Y(test_size), pred_seq(test_size), pred_par(test_size);

    generate_data(train_X, train_Y, train_size);
    generate_data(test_X, test_Y, test_size);  // We don’t use test_Y for accuracy here, only timing

    // Sequential
    auto start = high_resolution_clock::now();
    knn_sequential(train_X, train_Y, test_X, pred_seq, k);
    auto end = high_resolution_clock::now();
    cout << "\n=== Sequential KNN ===\n";
    cout << "Execution Time: " << duration_cast<milliseconds>(end - start).count() << " ms\n";

    // Parallel
    start = high_resolution_clock::now();
    knn_parallel(train_X, train_Y, test_X, pred_par, k);
    end = high_resolution_clock::now();
    cout << "\n=== Parallel KNN (OpenMP) ===\n";
    cout << "Execution Time: " << duration_cast<milliseconds>(end - start).count() << " ms\n";

    // Optionally: compare predictions match
    int match = 0;
    for (int i = 0; i < test_size; ++i)
        if (pred_seq[i] == pred_par[i]) match++;

    cout << "\nMatching Predictions: " << match << "/" << test_size << "\n";

    return 0;
}
