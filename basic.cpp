#include "eztimer/eztimer.hpp"

#include "CLI/App.hpp"
#include "CLI/Formatter.hpp"
#include "CLI/Config.hpp"

#include "scaled_ranks.h"

#include <random>
#include <vector>
#include <optional>
#include <iostream>
#include <limits>

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
    RankedVector negative_query, positive_query;
    std::vector<std::pair<int, double> > sparse_query;
    sparse_query.reserve(len);
    double zero_query;
    std::vector<double> dense_query(len);

    RankedVector negative_ref, positive_ref;
    std::vector<std::pair<int, double> > sparse_ref;
    sparse_ref.reserve(len);
    std::vector<int> sparse_ref_index;
    sparse_ref_index.reserve(len);
    std::vector<double> sparse_ref_value;
    sparse_ref_value.reserve(len);
    double zero_ref;
    std::vector<double> dense_ref(len);

    std::optional<double> result;

    // Setting up the simulation at each iteration.
    std::mt19937_64 rng(seed);
    std::normal_distribution<> normdist;
    std::uniform_real_distribution<> unifdist;

    eztimer::Options opt;
    opt.iterations = iterations;
    opt.setup = [&]() -> void {
        // Generating the query elements.
        negative_query.clear();
        positive_query.clear();
        for (int i = 0; i < len; ++i) {
            if (unifdist(rng) <= density) {
                double val = normdist(rng);
                if (val < 0) {
                    negative_query.emplace_back(val, i);
                } else if (val > 0) {
                    positive_query.emplace_back(val, i);
                }
            }
        }

        std::sort(negative_query.begin(), negative_query.end());
        std::sort(positive_query.begin(), positive_query.end());
        zero_query = scaled_ranks(len, negative_query, positive_query, sparse_query);
        std::sort(sparse_query.begin(), sparse_query.end());
        std::fill(dense_query.begin(), dense_query.end(), zero_query);
        for (const auto& sq : sparse_query) {
            dense_query[sq.first] = sq.second;
        }

        // Generating the reference elements.
        negative_ref.clear();
        positive_ref.clear();
        for (int i = 0; i < len; ++i) {
            if (unifdist(rng) <= density) {
                double val = normdist(rng);
                if (val < 0) {
                    negative_ref.emplace_back(val, i);
                } else if (val > 0) {
                    positive_ref.emplace_back(val, i);
                }
            }
        }

        std::sort(negative_ref.begin(), negative_ref.end());
        std::sort(positive_ref.begin(), positive_ref.end());
        zero_ref = scaled_ranks(len, negative_ref, positive_ref, sparse_ref);
        std::sort(sparse_ref.begin(), sparse_ref.end());

        sparse_ref_index.clear();
        sparse_ref_value.clear();
        dense_ref.resize(len);
        std::fill(dense_ref.begin(), dense_ref.end(), zero_ref);
        for (const auto& sr : sparse_ref) {
            sparse_ref_index.push_back(sr.first);
            sparse_ref_value.push_back(sr.second);
            dense_ref[sr.first] = sr.second;
        }

        result.reset();
    };

    // Setting up the functions.
    std::vector<std::function<double()> > funs;
    std::vector<std::string> names;

    names.push_back("dense-dense");
    funs.emplace_back([&]() -> double {
        double l2 = 0;
        for (int i = 0; i < len; ++i) {
            const double delta = dense_query[i] - dense_ref[i];
            l2 += delta * delta;
        }
        return l2;
    });

    names.push_back("sparse-dense-interleaved");
    funs.emplace_back([&]() -> double {
        int i = 0, j = 0;
        const int snum = sparse_query.size();
        double l2 = 0;

        while (j < snum) {
            const auto limit = sparse_query[j].first;
            for (; i < limit; ++i) {
                const auto delta = dense_ref[i] - zero_query;
                l2 += delta * delta;
            }
            const auto delta = dense_ref[i] - sparse_query[j].second;
            l2 += delta * delta;
            ++i;
            ++j;
        }

        for (; i < len; ++i) {
            const auto delta = dense_ref[i] - zero_query;
            l2 += delta * delta;
        }

        return l2;
    });

    names.push_back("dense-sparse-interleaved");
    funs.emplace_back([&]() -> double {
        int i = 0, j = 0;
        const int snum = sparse_ref_index.size();
        double l2 = 0;

        while (j < snum) {
            const auto limit = sparse_ref_index[j];
            for (; i < limit; ++i) {
                const auto delta = dense_query[i] - zero_ref;
                l2 += delta * delta;
            }
            const auto delta = dense_query[i] - sparse_ref_value[j];
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
    std::vector<double> buffer_ds(len);
    funs.emplace_back([&]() -> double {
        std::fill(buffer_ds.begin(), buffer_ds.end(), zero_ref);
        for (const auto& ss : sparse_ref) {
            buffer_ds[ss.first] = ss.second;
        }

        double val = 0;
        for (int i = 0; i < len; ++i) {
            const double delta = dense_query[i] - buffer_ds[i];
            val += delta * delta;
        }
        return val;
    });

    names.push_back("dense-sparse-densified2");
    std::vector<double> sd_mapping(len);
    funs.emplace_back([&]() -> double {
        const int num = sparse_ref_index.size();
        for (int i = 0; i < num; ++i) {
            sd_mapping[sparse_ref_index[i]] = sparse_ref_value[i] - zero_ref;
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

    names.push_back("dense-sparse-unstable");
    funs.emplace_back([&]() -> double {
        double l2 = 0;
        const int num = sparse_ref_index.size();
        for (int i = 0; i < num; ++i) {
            const double target = dense_query[sparse_ref_index[i]];
            const double ref = sparse_ref_value[i] - zero_ref;
            l2 += ref * (ref - 2 * target);
        }
        const double x2 = (sparse_query.empty() ? 0 : 0.25);
        return x2 + l2 - len * zero_ref * zero_ref;
    });

    names.push_back("sparse-sparse-interleaved");
    funs.emplace_back([&]() -> double {
        double l2 = 0;
        int i1 = 0, i2 = 0;
        int both = 0;
        const int snum1 = sparse_query.size();
        const int snum2 = sparse_ref_index.size();

        if (i1 < snum1 && i2 < snum2) { 
            while (1) {
                const auto idx1 = sparse_query[i1].first;
                const auto idx2 = sparse_ref_index[i2];
                if (idx1 < idx2) {
                    const double delta = sparse_query[i1].second - zero_ref;
                    l2 += delta * delta;
                    ++i1;
                    if (i1 == snum1) {
                        break;
                    }
                } else if (idx1 > idx2) {
                    const double delta = sparse_ref_value[i2] - zero_query;
                    l2 += delta * delta;
                    ++i2;
                    if (i2 == snum2) {
                        break;
                    }
                } else {
                    const double delta = sparse_query[i1].second - sparse_ref_value[i2];
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
            const double delta = sparse_ref_value[i2] - zero_query;
            l2 += delta * delta;
        }

        const double delta = zero_query - zero_ref;
        l2 += (len - snum1 - (snum2 - both)) * (delta * delta);
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
