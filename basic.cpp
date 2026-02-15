#include "eztimer/eztimer.hpp"

#include "CLI/App.hpp"
#include "CLI/Formatter.hpp"
#include "CLI/Config.hpp"

#include <random>
#include <vector>
#include <optional>
#include <iostream>

int main(int argc, char ** argv) {
    CLI::App app{"Sparse L2 calculation performance tests"};
    int len;
    app.add_option("-l,--length", len, "Length of the simulated vector")->default_val(1000);
    double density;
    app.add_option("-d,--density", density, "Density of non-zero elements in the simulated vector")->default_val(0.2);
    int iterations;
    app.add_option("-i,--iter", iterations, "Number of iterations")->default_val(100);
    unsigned long long seed;
    app.add_option("-s,--seed", seed, "Seed for the simulated data")->default_val(69);
    CLI11_PARSE(app, argc, argv);

    // Setting up all of the data structures.
    std::vector<double> dense_query(len);
    std::vector<std::pair<int, double> > sparse_query;
    double zero_query;
    sparse_query.reserve(len);

    std::vector<std::pair<int, double> > dense_ref, sparse_ref;
    double zero_ref;
    sparse_ref.reserve(len);

    std::vector<std::pair<int, double> > sparse_ref_sd, sparse_ref_ss;
    constexpr double empty = std::numeric_limits<double>::infinity();
    std::vector<double> ss_mapping(len);

    std::optional<double> result;

    // Setting up the simulation at each iteration.
    std::mt19937_64 rng(seed);
    std::normal_distribution<> normdist;
    std::uniform_real_distribution<> unifdist;

    eztimer::Options opt;
    opt.iterations = iterations;
    opt.setup = [&]() -> void {
        sparse_query.clear();
        zero_query = normdist(rng);
        for (int i = 0; i < len; ++i) {
            if (unifdist(rng) <= density) {
                const auto val = normdist(rng);
                sparse_query.emplace_back(i, val);
                dense_query[i] = val;
            } else {
                dense_query[i] = zero_query;
            }
        }

        sparse_ref.clear();
        dense_ref.clear();
        zero_ref = normdist(rng);
        for (int i = 0; i < len; ++i) {
            double val = zero_ref;
            if (unifdist(rng) <= density) {
                val = normdist(rng);
                sparse_ref.emplace_back(i, val);
            }
            dense_ref.emplace_back(i, val);
        }

        std::shuffle(dense_ref.begin(), dense_ref.end(), rng);
        std::shuffle(sparse_ref.begin(), sparse_ref.end(), rng);
        sparse_ref_sd.clear();
        sparse_ref_sd.insert(sparse_ref_sd.end(), sparse_ref.begin(), sparse_ref.end());
        sparse_ref_ss.clear();
        sparse_ref_ss.insert(sparse_ref_ss.end(), sparse_ref.begin(), sparse_ref.end());

        std::fill(ss_mapping.begin(), ss_mapping.end(), empty);
        for (const auto& ss : sparse_query) {
            ss_mapping[ss.first] = ss.second;
        }

        result.reset();
    };

    // Setting up the functions.
    std::vector<std::function<double()> > funs;
    std::vector<std::string> names;
    funs.reserve(7);
    names.reserve(7);

    names.push_back("dense-dense");
    std::vector<double> dense_mapping(len);
    funs.emplace_back([&]() -> double {
        for (const auto& dd : dense_ref) {
            dense_mapping[dd.first] = dd.second;
        }
        double l2 = 0;
        for (int i = 0; i < len; ++i) {
            const double delta = dense_query[i] - dense_mapping[i];
            l2 += delta * delta;
        }
        return l2;
    });

    names.push_back("sparse-dense-interleaved");
    std::vector<double> dense_mapping2(len);
    funs.emplace_back([&]() -> double {
        for (const auto& dd : dense_ref) {
            dense_mapping2[dd.first] = dd.second;
        }

        int i = 0, j = 0;
        const int snum = sparse_query.size();
        double l2 = 0;
        while (j < snum) {
            const auto limit = sparse_query[j].first;
            for (; i < limit; ++i) {
                const auto delta = dense_mapping2[i] - zero_query;
                l2 += delta * delta;
            }
            const auto delta = dense_mapping2[i] - sparse_query[j].second;
            l2 += delta * delta;
            ++i;
            ++j;
        }

        for (; i < len; ++i) {
            const auto delta = dense_mapping2[i] - zero_query;
            l2 += delta * delta;
        }

        return l2;
    });

    names.push_back("dense-sparse-interleaved");
    funs.emplace_back([&]() -> double {
        std::sort(sparse_ref_sd.begin(), sparse_ref_sd.end());
        int i = 0, j = 0;
        const int snum = sparse_ref_sd.size();
        double l2 = 0;

        while (j < snum) {
            const auto limit = sparse_ref_sd[j].first;
            for (; i < limit; ++i) {
                const auto delta = dense_query[i] - zero_ref;
                l2 += delta * delta;
            }
            const auto delta = dense_query[i] - sparse_ref_sd[j].second;
            l2 += delta * delta;
            ++i;
            ++j;
        }

        for (; i < len; ++i) {
            const auto delta = dense_query[i] - zero_ref;
            l2 += delta * delta;
        }

        return l2;
    });

    names.push_back("dense-sparse-densified");
    std::vector<double> buffer_dd(len);
    funs.emplace_back([&]() -> double {
        std::fill(buffer_dd.begin(), buffer_dd.end(), zero_ref);
        for (const auto& ss : sparse_ref) {
            buffer_dd[ss.first] = ss.second;
        }

        double val = 0;
        for (int i = 0; i < len; ++i) {
            const double delta = dense_query[i] - buffer_dd[i];
            val += delta * delta;
        }
        return val;
    });

    names.push_back("dense-sparse-densified2");
    std::vector<double> sd_mapping(len);
    funs.emplace_back([&]() -> double {
        for (const auto& ss : sparse_ref) {
            sd_mapping[ss.first] = ss.second - zero_ref;
        }

        double val = 0;
        for (int i = 0; i < len; ++i) {
            const double delta = (dense_query[i] - (sd_mapping[i] + zero_ref));
            val += delta * delta;
        }

        for (const auto& ss : sparse_ref) {
            sd_mapping[ss.first] = 0;
        }
        return val;
    });

    names.push_back("sparse-sparse-interleaved");
    funs.emplace_back([&]() -> double {
        std::sort(sparse_ref_ss.begin(), sparse_ref_ss.end());
        double l2 = 0;
        int i1 = 0, i2 = 0;
        int both = 0;
        const int snum1 = sparse_query.size();
        const int snum2 = sparse_ref_ss.size();

        if (i1 < snum1 && i2 < snum2) { 
            while (1) {
                const auto idx1 = sparse_query[i1].first;
                const auto idx2 = sparse_ref_ss[i2].first;
                if (idx1 < idx2) {
                    const double delta = sparse_query[i1].second - zero_ref;
                    l2 += delta * delta;
                    ++i1;
                    if (i1 == snum1) {
                        break;
                    }
                } else if (idx1 > idx2) {
                    const double delta = sparse_ref_ss[i2].second - zero_query;
                    l2 += delta * delta;
                    ++i2;
                    if (i2 == snum2) {
                        break;
                    }
                } else {
                    const double delta = sparse_query[i1].second - sparse_ref_ss[i2].second;
                    l2 += delta * delta;
                    ++i1;
                    ++i2;
                    ++both;
                    if (i1 == snum1 || i2 == snum2) {
                        break;
                    }
                }
            }
        }

        for (; i1 < snum1; ++i1) { 
            const double delta = sparse_query[i1].second - zero_ref;
            l2 += delta * delta;
        }
        for (; i2 < snum2; ++i2) { 
            const double delta = sparse_ref_ss[i2].second - zero_query;
            l2 += delta * delta;
        }

        const double delta = zero_query - zero_ref;
        l2 += (len - snum1 - (snum2 - both)) * (delta * delta);
        return l2;
    });

    names.push_back("sparse-sparse-remapped");
    funs.emplace_back([&]() -> double {
        double l2 = 0;
        int both = 0;
        for (const auto& ss : sparse_ref) {
            double& target = ss_mapping[ss.first];
            if (target == empty) {
                const double delta = ss.second - zero_query;
                l2 += delta * delta;
            } else {
                const double delta = ss.second - target;
                l2 += delta * delta;
                target = empty;
                ++both;
            }
        }

        for (const auto& ss : sparse_query) {
            double& target = ss_mapping[ss.first];
            if (target != empty) {
                const double delta = target - zero_ref;
                l2 += delta * delta;
            } else {
                target = ss.second;
            }
        }

        const double delta = zero_query - zero_ref;
        l2 += ((len - sparse_query.size()) - (sparse_ref.size() - both)) * delta * delta;
        return l2;
    });

    // Performing the iterations.
    auto res = eztimer::time<double>(
        funs,
        [&](const double& res, std::size_t i) -> void {
            if (result.has_value()) {
                if (std::abs(*result - res) / res > 1e-8) {
                    std::cout << *result << "\t" << res << "\t" << names[i] << std::endl;
                    throw std::runtime_error("oops that's not right");
                }
            } else {
                result = res;
            }
        },
        opt
    );

    for (std::size_t n = 0; n < names.size(); ++n) {
        std::string nn = names[n];
        nn.resize(32, ' ');
        const double mu = res[n].mean.count(); 
        const double se = res[n].sd.count() / std::sqrt(res[n].times.size());
        std::cout << nn << ": " << mu << " ± " << (se / mu * 100) << " %" << std::endl;
    }

    return 0;
}

