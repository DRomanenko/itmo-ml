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
    size_t k1, k2, number_objects;
    cin >> k1
        >> k2
        >> number_objects;
    double ans = 0;
    map<pair<double, double>, double> m;
    vector<double> prob1(k1, 0), prob2(k2, 0);
    for (size_t i = 0; i < number_objects; ++i) {
        double x, y;
        cin >> x >> y;
        prob1[x - 1] += 1.0 / number_objects;
        prob2[y - 1] += 1.0 / number_objects;
        if (m.find({x, y}) == m.end())
            m[{x, y}] = 0;
        m[{x, y}] += 1;
    }
    for (auto item : m) {
        double help = prob1[item.first.first - 1] * prob2[item.first.second - 1] * number_objects;
        ans += (item.second - help) / help * (item.second - help) - help;
    }
    cout << setprecision(7) << ans + number_objects << '\n';
    return 0;
}