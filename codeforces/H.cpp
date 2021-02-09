#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include <iomanip>
#include <numeric>
#include <cmath>

using namespace std;

#define ll long long

signed main() {
    // freopen((FILE_NAME + ".in").c_str(), "r", stdin);
    // freopen((FILE_NAME + ".out").c_str(), "w", stdout);
    std::ios_base::sync_with_stdio(false), std::cin.tie(nullptr), std::cout.tie(nullptr);
    size_t m;
    cin >> m;
    vector<long long> f(1 << m);
    for (size_t i = 0; i < (1 << m); ++i)
        cin >> f[i];
    long long sum = accumulate(f.begin(), f.end(), 0LL);
    if (sum) {
        cout << "2\n" << sum << " 1\n";
        for (size_t i = 0; i < (1 << m); ++i) {
            if (f[i] == 1) {
                vector<double> help;
                size_t save = i;
                for (size_t q = 0; q < m; ++q) {
                    help.push_back(save % 2);
                    cout << (save % 2 ? 1 : -1) << ' ';
                    cout.flush();
                    save /= 2;
                }
                cout << 0.5 - accumulate(help.begin(), help.end(), 0.0) << '\n';
                cout.flush();
            }
        }
        for (size_t i = 0; i < sum; ++i)
            cout << "1 ";
        cout << "-0.5\n";
    } else {
        cout << "1\n1\n";
        for (size_t i = 0; i < m; ++i)
            cout << "0 ";
        cout << "-0.5\n";
    }
    return 0;
}