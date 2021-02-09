#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include <iomanip>

using namespace std;

#define ll long long

signed main() {
    // freopen((FILE_NAME + ".in").c_str(), "r", stdin);
    // freopen((FILE_NAME + ".out").c_str(), "w", stdout);
    std::ios_base::sync_with_stdio(false), std::cin.tie(nullptr), std::cout.tie(nullptr);
    size_t k, number_objects;
    cin >> k >> number_objects;
    vector<vector<ll>> matrix(number_objects, vector<ll>(2));
    for (size_t i = 0; i < number_objects; ++i)
        cin >> matrix[i][0] >> matrix[i][1];
    vector<vector<ll>> classes(k, vector<ll>());
    vector<ll> qty_classes(vector<ll>(k, 0)), sum_classes(vector<ll>(k, 0));
    for (size_t i = 0; i < number_objects; ++i)
        classes[matrix[i][1] - 1].push_back(matrix[i][0]);
    ll inclass_distance = 0, class_distance = 0;
    sort(matrix.begin(), matrix.end());
    for (const auto& help : classes) {
        vector<ll> values = help;
        sort(values.begin(), values.end());
        ll sum = 0;
        for (size_t i = 0 ; i < values.size(); ++i) {
            sum += values[i];
            inclass_distance += ((ll) (i + 1) * values[i] - sum);
        }
    }
    ll sum = 0;
    for (size_t i = 0; i < number_objects; ++i) {
        sum += matrix[i][0];
        qty_classes[matrix[i][1] - 1] += 1;
        sum_classes[matrix[i][1] - 1] += matrix[i][0];
        class_distance += matrix[i][0] * (i + 1 - qty_classes[matrix[i][1] - 1]) + sum_classes[matrix[i][1] - 1] - sum;
    }
    cout << 2 * inclass_distance << '\n' << 2 * class_distance << '\n';
    return 0;
}
