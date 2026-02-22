# Sparse/dense L2 calculations

## Overview

[**singlepp**](https://github.com/SingleR-inc/singlepp) can construct reference objects from both dense and sparse matrices.
The reference objects store the values of each reference profile in sorted order, ready for calculation of scaled ranks during fine-tuning.
(For sparse matrices, we store the negative and positives values separately, omitting all of the zeros.)
Spearman's correlation is derived from the L2 norm of the difference between the vectors of scaled ranks of the query and reference profiles. 
The question is, what is the fastest way to compute this L2 norm?

We consider two variations of the issue.
The "basic" approach involves calculating L2 norms when the reference's scaled ranks have already been computed.
In the sparse case, this allows us to assume that the scaled ranks are sorted by position rather than value.
By comparison, the "fine-tune" approach requires the calculation of scaled ranks for each reference profile before the L2 norm.
Ranks are not sorted by position after scaling, which has some performance implications due to non-contiguous memory access speed. 

To run these timings, use the usual CMake process.
This produces two binaries - `basic` and `fine_tune` - to measure performance under the two scenarios described above.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## Algorithms

`dense-dense`: when both the query and reference are dense, we compute the L2 norm by iterating over both arrays at once and summing the squared differences.
In the fine-tuning case, the iteration is not contiguous over the query array as the reference values are not sorted by position.

When the query is sparse and the reference is dense, we have several choices:

- The simplest is to just convert the query into a dense buffer, which is the same as the `dense-dense` calculation.
  Such operations on the query are considered to be cheap as it only has to be done once.
- `sparse-dense-interleaved`: another option is to interleave the summations. 
  Assuming that the query's non-zero values are sorted by position,
  we compute the sum of squared differences between the query array against the scaled rank of the reference's zero value,
  all the way up to the next non-zero reference value.
  We then add the squared difference to that non-zero reference value and continue until completion.
  This avoids populating a separate dense array for the query.

When the query is dense and the reference is sparse, we have several choices:

- `dense-sparse-interleaved`: effectively a duplicate of `sparse-dense-interleaved` in the basic case.
  For fine-tuning, we need to resort the reference's non-zero values by index first. 
  This is because the reference's scaled rank vector would be originally sorted by value.
- `dense-sparse-densified`: we convert the reference's non-zero values into a dense array.
  This requires one memory access to fill the dense buffer with the scaled rank of the reference's zero value,
  and then another few accesses to populate all of the non-zero values.
  The summation then proceeds as described for `dense-dense`.
- `dense-sparse-densified2`: a variation of `dense-sparse-densified`.
  We populate the dense array with `x2 := x - zero_val` where `x` is the rank of each non-zero value and `zero_rank` is the scaled rank of the reference's zero value.
  Then the summation is performed on the squared value of `y - (x2 + zero_val)` across the array.
  We then reset the dense array by setting the values of the array to zero at the non-zero indices.
  This saves us a `memset` across the entire array at the cost of an extra summation during the summation loop.
- `dense-sparse-unstable`: we iterate across the reference's sparse values,
  computing the sum of `x2 * (x2 - 2 * y)` where `x2` is as defined for `dense-sparse-densified2`.
  The L2 norm is then calculated as `0.25 + S - n * zero_val^2` where `S` is the sum of the aforementioned product and `n` is the total number of genes.
  This represents an alternative formulation of the L2 norm that sacrifices some numerical stability for fast iteration over a sparse vector.

When both the query and reference are sparse, we can do:

- `sparse-sparse-interleaved`: sort the reference's non-zero values by index,
  and then walk through the reference/query non-zero values by increasing index to perform the summation.
  For fine-tuning, we need to resort the reference's non-zero values by index first. 
- We could also just convert the query to a dense array beforehand, which boils down to any of the `dense-sparse-*` choices.

## Results

For an Intel i7-8850H running Ubuntu Linux, we get:

```bash
$ ./build/basic -d 0.2 -l 1000
dense-dense                     : 4.01641e-06 ± 1.13607 %
sparse-dense-interleaved        : 9.22126e-06 ± 1.18531 %
dense-sparse-interleaved        : 8.91404e-06 ± 1.20366 %
dense-sparse-densified          : 5.02495e-06 ± 1.14064 %
dense-sparse-densified2         : 4.73814e-06 ± 1.14977 %
dense-sparse-unstable           : 9.3097e-07 ± 1.30051 %
sparse-sparse-interleaved       : 6.79047e-06 ± 1.22709 %

$ ./build/basic -d 0.2 -l 10000
dense-dense                     : 9.69449e-06 ± 0.587642 %
sparse-dense-interleaved        : 2.20291e-05 ± 0.272301 %
dense-sparse-interleaved        : 2.16533e-05 ± 0.958304 %
dense-sparse-densified          : 1.34323e-05 ± 2.12897 %
dense-sparse-densified2         : 1.31565e-05 ± 2.31519 %
dense-sparse-unstable           : 2.25133e-06 ± 5.5954 %
sparse-sparse-interleaved       : 1.65123e-05 ± 0.254594 %

$ ./build/basic -d 0.05 -l 1000
dense-dense                     : 9.6354e-07 ± 0.084539 %
sparse-dense-interleaved        : 1.02164e-06 ± 0.320924 %
dense-sparse-interleaved        : 1.01362e-06 ± 0.314254 %
dense-sparse-densified          : 1.16051e-06 ± 0.135375 %
dense-sparse-densified2         : 1.01305e-06 ± 0.131222 %
dense-sparse-unstable           : 7.99e-08 ± 0.95656 %
sparse-sparse-interleaved       : 4.2961e-07 ± 1.07232 %

$ ./build/basic -d 0.5 -l 1000
dense-dense                     : 1.0031e-06 ± 0.904484 %
sparse-dense-interleaved        : 3.8956e-06 ± 7.87659 %
dense-sparse-interleaved        : 3.69909e-06 ± 4.96098 %
dense-sparse-densified          : 1.49956e-06 ± 2.67721 %
dense-sparse-densified2         : 1.46455e-06 ± 1.1693 %
dense-sparse-unstable           : 5.3749e-07 ± 1.80657 %
sparse-sparse-interleaved       : 4.23637e-06 ± 3.83337 %
```

For fine-tuning:

```bash
$ ./build/fine_tune -d 0.2 -l 1000
dense-dense                     : 9.59304e-06 ± 3.79061 %
sparse-dense-interleaved        : 1.53516e-05 ± 3.82049 %
dense-sparse-interleaved        : 2.85492e-05 ± 3.92428 %
dense-sparse-densified          : 8.15236e-06 ± 3.80569 %
dense-sparse-densified2         : 8.35445e-06 ± 4.15181 %
dense-sparse-unstable           : 4.07952e-06 ± 3.827 %
sparse-sparse-interleaved       : 2.7533e-05 ± 4.08709 %

$ ./build/fine_tune -d 0.2 -l 10000
dense-dense                     : 2.64288e-05 ± 1.30079 %
sparse-dense-interleaved        : 4.13857e-05 ± 1.20586 %
dense-sparse-interleaved        : 0.000107717 ± 0.621039 %
dense-sparse-densified          : 2.37046e-05 ± 1.21688 %
dense-sparse-densified2         : 2.36759e-05 ± 1.22765 %
dense-sparse-unstable           : 1.10571e-05 ± 1.09754 %
sparse-sparse-interleaved       : 0.00010189 ± 0.656838 %

$ ./build/fine_tune -d 0.05 -l 1000
dense-dense                     : 1.98422e-06 ± 0.220243 %
sparse-dense-interleaved        : 2.34852e-06 ± 6.81117 %
dense-sparse-interleaved        : 2.14402e-06 ± 1.37152 %
dense-sparse-densified          : 1.46171e-06 ± 0.269653 %
dense-sparse-densified2         : 1.25136e-06 ± 0.382146 %
dense-sparse-unstable           : 3.1718e-07 ± 1.22646 %
sparse-sparse-interleaved       : 1.58329e-06 ± 1.8076 %

$ ./build/fine_tune -d 0.5 -l 1000
dense-dense                     : 3.88538e-06 ± 1.95473 %
sparse-dense-interleaved        : 6.99965e-06 ± 2.00358 %
dense-sparse-interleaved        : 2.17285e-05 ± 1.98859 %
dense-sparse-densified          : 4.02357e-06 ± 3.8273 %
dense-sparse-densified2         : 4.06212e-06 ± 2.1938 %
dense-sparse-unstable           : 2.93423e-06 ± 3.22236 %
sparse-sparse-interleaved       : 2.26536e-05 ± 2.11379 %
```

Some comments:

- The interleaving approaches perform poorly.
  I would guess that the interleaving involves so many branch mispredictions that it's just faster to do more memory accesses and create a dense array. 
- `dense-sparse-unstable` is consistently good and indicates that storing sparse references is beneficial.
  I would wager that some numerical inaccuracy is acceptable here as distances close to zero would result in correlations close to 1, at which point small errors can be ignored.
  It is probably a good idea to clamp the correlations to [-1, 1] to avoid egregiously wrong results, e.g., correlations > 1.
- There is no benefit from the query being sparse, which allows us to simplify the **singlepp** code considerably. 
