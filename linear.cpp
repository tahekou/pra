#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

// Generate synthetic data: Y = 2x + noise
void generate_data(vector<double>& X, vector<double>& Y, int n) {
    srand(time(0));
    for (int i = 0; i < n; ++i) {
        double x = rand() % 100;
        double noise = (rand() % 100 - 50) / 100.0;
        X.push_back(x);
        Y.push_back(2 * x + noise); // True line: Y = 2x + noise
    }
}

// Sequential Gradient Descent
void linear_regression_seq(const vector<double>& X, const vector<double>& Y, double& w, double& b, double lr, int epochs) {
    int n = X.size();
    for (int e = 0; e < epochs; ++e) {
        double dw = 0, db = 0;
        for (int i = 0; i < n; ++i) {
            double Y_pred = w * X[i] + b;
            double error = Y_pred - Y[i];
            dw += error * X[i];
            db += error;
        }
        dw /= n;
        db /= n;
        w -= lr * dw;
        b -= lr * db;
    }
}

// Parallel Gradient Descent using OpenMP
void linear_regression_parallel(const vector<double>& X, const vector<double>& Y, double& w, double& b, double lr, int epochs) {
    int n = X.size();
    for (int e = 0; e < epochs; ++e) {
        double dw = 0, db = 0;
        #pragma omp parallel for reduction(+:dw, db)
        for (int i = 0; i < n; ++i) {
            double Y_pred = w * X[i] + b;
            double error = Y_pred - Y[i];
            dw += error * X[i];
            db += error;
        }
        dw /= n;
        db /= n;
        w -= lr * dw;
        b -= lr * db;
    }
}

int main() {
    int n = 100000;
    vector<double> X, Y;
    generate_data(X, Y, n);

    double w_seq = 0, b_seq = 0;
    double w_par = 0, b_par = 0;
    double lr = 0.0001;
    int epochs = 100;

    // Time Sequential
    auto start = high_resolution_clock::now();
    linear_regression_seq(X, Y, w_seq, b_seq, lr, epochs);
    auto end = high_resolution_clock::now();
    cout << "\n=== Sequential Linear Regression ===\n";
    cout << "w = " << w_seq << ", b = " << b_seq << endl;
    cout << "Time: " << duration_cast<milliseconds>(end - start).count() << " ms\n";

    // Time Parallel
    start = high_resolution_clock::now();
    linear_regression_parallel(X, Y, w_par, b_par, lr, epochs);
    end = high_resolution_clock::now();
    cout << "\n=== Parallel Linear Regression (OpenMP) ===\n";
    cout << "w = " << w_par << ", b = " << b_par << endl;
    cout << "Time: " << duration_cast<milliseconds>(end - start).count() << " ms\n";

    return 0;
}



#include <iostream>
#include <omp.h>
using namespace std;

int main() {
    const int n = 1024;
    float x[n], y[n];

    // Normalize input to prevent overflow
    // here we are creating our large data of 1024 points
    for (int i = 0; i < n; ++i) {
        x[i] = (i + 1) / 1024.0f;  // Normalized x
        y[i] = 2 * x[i] + 3;       // y = 2x + 3
    }

    float w = 0, b = 0, lr = 0.01;
    float dw, db;

    for (int epoch = 0; epoch < 2000; ++epoch) {
        dw = 0.0f;
        db = 0.0f;

        // Parallel gradient computation
        #pragma omp parallel for reduction(+:dw, db)
        for (int i = 0; i < n; ++i) {
            float y_pred = w * x[i] + b;
            float error = y_pred - y[i];
            dw += 2 * x[i] * error / n;
            db += 2 * error / n;
        }

        w -= lr * dw;
        b -= lr * db;

        if(epoch%100==0) {
            cout << "Epoch " << epoch << ": w = " << w << ", b = " << b << endl;
        }
    }

    cout << "Learned w: " << w << ", b: " << b << endl;
    return 0;
}
