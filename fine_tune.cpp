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

    RankedVector negative_ref, positive_ref, full_ref;
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
        full_ref.clear();
        for (int i = 0; i < len; ++i) {
            if (unifdist(rng) <= density) {
                double val = normdist(rng);
                if (val < 0) {
                    negative_ref.emplace_back(val, i);
                } else if (val > 0) {
                    positive_ref.emplace_back(val, i);
                }
                full_ref.emplace_back(val, i);
            } else {
                full_ref.emplace_back(0, i);
            }
        }

        std::sort(negative_ref.begin(), negative_ref.end());
        std::sort(positive_ref.begin(), positive_ref.end());
        std::sort(full_ref.begin(), full_ref.end());

        result.reset();
    };

    // Setting up the functions.
    std::vector<std::function<double()> > funs;
    std::vector<std::string> names;

    names.push_back("dense-dense");
    std::vector<double> dd_buffer(len);
    funs.emplace_back([&]() -> double {
        double l2 = 0;
        scaled_ranks(
            len,
            full_ref,
            dd_buffer.data(),
            [&](const int i, const double val) -> void {
                const double delta = dense_query[i] - val;
                l2 += delta * delta;
            }
        );
        return l2;
    });

    names.push_back("sparse-dense-interleaved");
    std::vector<double> sd_buffer(len);
    funs.emplace_back([&]() -> double {
        scaled_ranks(
            len,
            full_ref,
            sd_buffer.data(),
            [&](const int i, const double val) -> void {
                sd_buffer[i] = val;
            }
        );

        int i = 0, j = 0;
        const int snum = sparse_query.size();
        double l2 = 0;

        while (j < snum) {
            const auto limit = sparse_query[j].first;
            for (; i < limit; ++i) {
                const auto delta = sd_buffer[i] - zero_query;
                l2 += delta * delta;
            }
            const auto delta = sd_buffer[i] - sparse_query[j].second;
            l2 += delta * delta;
            ++i;
            ++j;
        }

        for (; i < len; ++i) {
            const auto delta = sd_buffer[i] - zero_query;
            l2 += delta * delta;
        }

        return l2;
    });

    names.push_back("dense-sparse-interleaved");
    std::vector<std::pair<int, double> > dsi_tmp;
    dsi_tmp.reserve(len);
    funs.emplace_back([&]() -> double {
        double zero_ref;
        scaled_ranks(
            len,
            negative_ref,
            positive_ref,
            dsi_tmp,
            [&](const double zval) -> void {
                zero_ref = zval;
            },
            [&](std::pair<int, double>& pair, const double val) -> void {
                pair.second = val;
            }
        );
        std::sort(dsi_tmp.begin(), dsi_tmp.end());

        int i = 0, j = 0;
        const int snum = dsi_tmp.size();
        double l2 = 0;

        while (j < snum) {
            const auto limit = dsi_tmp[j].first;
            for (; i < limit; ++i) {
                const auto delta = dense_query[i] - zero_ref;
                l2 += delta * delta;
            }
            const auto delta = dense_query[i] - dsi_tmp[j].second;
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
    std::vector<std::pair<int, double> > dsd_tmp;
    dsd_tmp.reserve(len);
    std::vector<double> dsd_buffer(len);
    funs.emplace_back([&]() -> double {
        scaled_ranks(
            len,
            negative_ref,
            positive_ref,
            dsd_tmp,
            [&](const double zval) -> void {
                std::fill(dsd_buffer.begin(), dsd_buffer.end(), zval);
            },
            [&](std::pair<int, double>& pair, const double val) -> void {
                dsd_buffer[pair.first] = val;
            }
        );

        double val = 0;
        for (int i = 0; i < len; ++i) {
            const double delta = dense_query[i] - dsd_buffer[i];
            val += delta * delta;
        }
        return val;
    });

    names.push_back("dense-sparse-densified2");
    std::vector<std::pair<int, double> > dsd2_tmp;
    dsd2_tmp.reserve(len);
    std::vector<double> dsd2_mapping(len);
    funs.emplace_back([&]() -> double {
        double zero_ref;
        scaled_ranks(
            len,
            negative_ref,
            positive_ref,
            dsd2_tmp,
            [&](const double zval) -> void {
                zero_ref = zval;
            },
            [&](std::pair<int, double>& pair, const double val) -> void {
                dsd2_mapping[pair.first] = val - zero_ref;
            }
        );

        double val = 0;
        for (int i = 0; i < len; ++i) {
            const double delta = (dense_query[i] - (dsd2_mapping[i] + zero_ref));
            val += delta * delta;
        }

        for (const auto& ss : dsd2_tmp) {
            dsd2_mapping[ss.first] = 0;
        }
        return val;
    });

    names.push_back("sparse-sparse-interleaved");
    std::vector<std::pair<int, double> > ssi_tmp;
    ssi_tmp.reserve(len);
    funs.emplace_back([&]() -> double {
        double zero_ref;
        scaled_ranks(
            len,
            negative_ref,
            positive_ref,
            ssi_tmp,
            [&](const double zval) -> void {
                zero_ref = zval;
            },
            [&](std::pair<int, double>& pair, const double val) -> void {
                pair.second = val;
            }
        );
        std::sort(ssi_tmp.begin(), ssi_tmp.end());

        double l2 = 0;
        int i1 = 0, i2 = 0;
        int both = 0;
        const int snum1 = sparse_query.size();
        const int snum2 = ssi_tmp.size();

        if (i1 < snum1 && i2 < snum2) { 
            while (1) {
                const auto idx1 = sparse_query[i1].first;
                const auto idx2 = ssi_tmp[i2].first;
                if (idx1 < idx2) {
                    const double delta = sparse_query[i1].second - zero_ref;
                    l2 += delta * delta;
                    ++i1;
                    if (i1 == snum1) {
                        break;
                    }
                } else if (idx1 > idx2) {
                    const double delta = ssi_tmp[i2].second - zero_query;
                    l2 += delta * delta;
                    ++i2;
                    if (i2 == snum2) {
                        break;
                    }
                } else {
                    const double delta = sparse_query[i1].second - ssi_tmp[i2].second;
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
            const double delta = ssi_tmp[i2].second - zero_query;
            l2 += delta * delta;
        }

        const double delta = zero_query - zero_ref;
        l2 += (len - snum1 - (snum2 - both)) * (delta * delta);
        return l2;
    });

    names.push_back("any-sparse-unstable");
    std::vector<std::pair<int, double> > asu_tmp;
    asu_tmp.reserve(len);
    funs.emplace_back([&]() -> double {
        double l2 = 0;
        const double x2 = (sparse_query.empty() ? 0 : 0.25);

        double zero_ref;
        scaled_ranks(
            len,
            negative_ref,
            positive_ref,
            asu_tmp,
            [&](const double zval) -> void {
                zero_ref = zval;
            },
            [&](std::pair<int, double>& pair, const double val) -> void {
                const double target = dense_query[pair.first];
                const double ref = val - zero_ref;
                l2 += ref * (ref - 2 * target);
            }
        );

        return x2 + l2 - len * zero_ref * zero_ref;
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
                std::cout << res << "\t" << names[i] << std::endl;
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
