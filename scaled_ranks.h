#ifndef SCALED_RANKS_H
#define SCALED_RANKS_H

#include <algorithm>
#include <vector>
#include <cmath>
#include <type_traits>
#include <cassert>

typedef std::vector<std::pair<double, int> > RankedVector;

template<class Process_>
void scaled_ranks(
    const int num_markers,
    const RankedVector& collected,
    double* buffer,
    Process_ process
) { 
    if (num_markers == 0) {
        return;
    }

    const double center_rank = static_cast<double>(num_markers - 1) / static_cast<double>(2); 
    double sum_squares = 0;

    // Computing tied ranks. 
    int cur_rank = 0;
    auto cIt = collected.begin();
    const auto cEnd = collected.end();

    while (cIt != cEnd) {
        auto copy = cIt;
        do {
            ++copy;
        } while (copy != cEnd && copy->first == cIt->first);

        const double jump = copy - cIt;
        const double mean_rank = cur_rank + (jump - 1) / static_cast<double>(2) - center_rank;
        sum_squares += mean_rank * mean_rank * jump;

        while (cIt != copy) {
            buffer[cIt->second] = mean_rank;
            ++cIt;
        }

        cur_rank += jump;
    }

    // Special behaviour for no-variance cells; these are left as all-zero scaled ranks.
    if (sum_squares == 0) {
        for (int i = 0; i < num_markers; ++i) {
            process(i, 0.0);
        }
    } else {
        const double denom = 0.5 / std::sqrt(sum_squares);
        for (int i = 0; i < num_markers; ++i) {
            process(i, buffer[i] * denom);
        }
    }
}

template<class ZeroProcess_, class Process_>
void scaled_ranks(
    const int num_markers,
    const RankedVector& negative,
    const RankedVector& positive,
    std::vector<std::pair<int, double> >& buffer,
    ZeroProcess_ zero,
    Process_ process
) {
    buffer.clear();
    if (num_markers == 0) {
        zero(0);
        return;
    }

    const double center_rank = static_cast<double>(num_markers - 1) / static_cast<double>(2); 
    double sum_squares = 0;

    // Computing tied ranks: before, at, and after zero.
    int cur_rank = 0;
    auto nIt = negative.begin();
    const auto negative_end = negative.end();
    while (nIt != negative_end) {
        auto copy = nIt;
        do {
            ++copy;
        } while (copy != negative_end && copy->first == nIt->first);

        const double jump = copy - nIt;
        const double mean_rank = cur_rank + static_cast<double>(jump - 1) / static_cast<double>(2) - center_rank;
        sum_squares += mean_rank * mean_rank * jump;

        while (nIt != copy) {
            buffer.emplace_back(nIt->second, mean_rank);
            ++nIt;
        }

        cur_rank += jump;
    }

    int num_zero = num_markers - negative.size() - positive.size();
    double zero_rank = 0; 
    if (num_zero) {
        zero_rank = cur_rank + static_cast<double>(num_zero - 1) / static_cast<double>(2) - center_rank;
        sum_squares += zero_rank * zero_rank * num_zero;
        cur_rank += num_zero;
    }

    auto pIt = positive.begin();
    const auto positive_end = positive.end();
    while (pIt != positive_end) {
        auto copy = pIt;
        do {
            ++copy;
        } while (copy != positive_end && copy->first == pIt->first);

        const double jump = copy - pIt;
        const double mean_rank = cur_rank + static_cast<double>(jump - 1) / static_cast<double>(2) - center_rank;
        sum_squares += mean_rank * mean_rank * jump;

        while (pIt != copy) {
            buffer.emplace_back(pIt->second, mean_rank);
            ++pIt;
        }

        cur_rank += jump;
    }

    // Special behaviour for no-variance cells; these are left as all-zero scaled ranks.
    if (sum_squares == 0) {
        zero(0);
        buffer.clear();
        return;
    }

    const double denom = 0.5 / std::sqrt(sum_squares);
    zero(zero_rank * denom);
    for (auto& nz : buffer) {
        process(nz, nz.second * denom);
    }
}

inline double scaled_ranks(
    const int num_markers,
    const RankedVector& negative,
    const RankedVector& positive,
    std::vector<std::pair<int, double> >& buffer
) {
    double output;
    scaled_ranks(
        num_markers,
        negative,
        positive,
        buffer,
        [&](const double zval) -> void {
            output = zval;
        },
        [&](std::pair<int, double>& pair, const double val) -> void {
            pair.second = val;
        }
    );
    return output;
}

#endif
