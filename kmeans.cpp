#include <bits/stdc++.h>
#include <omp.h>
using namespace std;
using namespace std::chrono;

struct Point {
    int x;
    int y;
    int c; // cluster assignment
};

// Euclidean distance
double cal_dist(Point a, Point b) {
    return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}

// Sequential K-Means
void kmeansseq(vector<Point>& points, vector<Point>& centroids, int k, int n) {
    bool isChanged = true;

    while (isChanged) {
        isChanged = false;

        // Assignment step
        for (int i = 0; i < n; i++) {
            double mindist = DBL_MAX;
            int bestCluster = -1;

            for (int j = 0; j < k; j++) {
                double dist = cal_dist(points[i], centroids[j]);
                if (dist < mindist) {
                    mindist = dist;
                    bestCluster = j;
                }
            }

            if (points[i].c != bestCluster) {
                points[i].c = bestCluster;
                isChanged = true;
            }
        }

        // Update step
        vector<double> sumx(k, 0), sumy(k, 0);
        vector<int> count(k, 0);

        for (int i = 0; i < n; i++) {
            int cluster = points[i].c;
            sumx[cluster] += points[i].x;
            sumy[cluster] += points[i].y;
            count[cluster]++;
        }

        for (int j = 0; j < k; j++) {
            if (count[j] > 0) {
                centroids[j].x = sumx[j] / count[j];
                centroids[j].y = sumy[j] / count[j];
            }
        }
    }
}

// Parallel K-Means using OpenMP
void kmeanspar(vector<Point>& points, vector<Point>& centroids, int k, int n) {
    bool changed = true;

    while (changed) {
        changed = false;

        vector<int> new_clusters(n);

        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            double mindist = DBL_MAX;
            int bestCluster = -1;
            for (int j = 0; j < k; j++) {
                double dist = cal_dist(points[i], centroids[j]);
                if (dist < mindist) {
                    mindist = dist;
                    bestCluster = j;
                }
            }
            new_clusters[i] = bestCluster;
        }

        // Apply new clusters and check for changes
        for (int i = 0; i < n; i++) {
            if (points[i].c != new_clusters[i]) {
                changed = true;
                points[i].c = new_clusters[i];
            }
        }

        // Update centroids
        vector<double> sumx(k, 0), sumy(k, 0);
        vector<int> count(k, 0);

        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            int c = points[i].c;
            #pragma omp atomic
            sumx[c] += points[i].x;
            #pragma omp atomic
            sumy[c] += points[i].y;
            #pragma omp atomic
            count[c]++;
        }

        #pragma omp parallel for
        for (int j = 0; j < k; j++) {
            if (count[j] > 0) {
                centroids[j].x = sumx[j] / count[j];
                centroids[j].y = sumy[j] / count[j];
            }
        }
    }
}

int main() {
    int n;
    cout << "Enter number of points: ";
    cin >> n;

    int k = 3; // number of clusters
    vector<Point> points(n);
    vector<Point> centroids(k);
    srand(time(0));

    for (int i = 0; i < n; i++) {
        points[i].x = rand() % 1000;
        points[i].y = rand() % 1000;
        points[i].c = -1;
    }

    // Initialize centroids randomly from points
    for (int i = 0; i < k; i++) {
        centroids[i] = points[rand() % n];
    }

    vector<Point> points_copy = points;
    vector<Point> centroids_copy = centroids;

    // Sequential K-Means
    auto start = high_resolution_clock::now();
    kmeansseq(points, centroids, k, n);
    auto end = high_resolution_clock::now();
    cout << "Sequential K-Means time: " << duration_cast<milliseconds>(end - start).count() << " ms" << endl;

    // Parallel K-Means (restart with same data)
    start = high_resolution_clock::now();
    kmeanspar(points_copy, centroids_copy, k, n);
    end = high_resolution_clock::now();
    cout << "Parallel K-Means time: " << duration_cast<milliseconds>(end - start).count() << " ms" << endl;

    return 0;
}
