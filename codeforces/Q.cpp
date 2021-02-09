#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include <iomanip>
#include <cmath>

using namespace std;

#define ll long long

signed main() {
    // freopen((FILE_NAME + ".in").c_str(), "r", stdin);
    // freopen((FILE_NAME + ".out").c_str(), "w", stdout);
    std::ios_base::sync_with_stdio(false), std::cin.tie(nullptr), std::cout.tie(nullptr);
    size_t kx, ky, number_objects;
    cin >> kx
        >> ky
        >> number_objects;
    double ans = 0;
    map<pair<double, double>, double> m;
    vector<double> prob1(kx, 0), h(kx, 0);
    for (size_t i = 0; i < number_objects; ++i) {
        double x, y;
        cin >> x >> y;
        x -= 1, y -= 1;
        prob1[x] += 1.0;
        if (m.find({x, y}) == m.end())
            m[{x, y}] = 0;
        m[{x, y}] += 1;
    }
    for (auto item : m)
        h[item.first.first] -= log(m[item.first] / prob1[item.first.first]) * m[item.first] / prob1[item.first.first];
    for (size_t i = 0; i < kx; ++i)
        ans += prob1[i] * h[i] / number_objects;
    cout << setprecision(7) << ans << '\n';
    return 0;
}