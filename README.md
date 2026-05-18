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
- `sparse-dense-unstable-sorted`: same as `dense-sparse-unstable` except that the sparse vector is not sorted by the index.
  This might occur if the sparse vector is derived from the query (in which case we can't sort ahead of time) and we're comparing to a dense reference.

When both the query and reference are sparse, we can do:

- `sparse-sparse-interleaved`: sort the reference's non-zero values by index,
  and then walk through the reference/query non-zero values by increasing index to perform the summation.
  For fine-tuning, we need to resort the reference's non-zero values by index first. 
- We could also just convert the query to a dense array beforehand, which boils down to any of the `dense-sparse-*` choices.

## Results

For an Intel i7-8850H running Ubuntu Linux, we get:

```bash
$ ./build/basic -d 0.2 -l 10000
dense-dense                     : 9.67716e-06 ± 0.35449 %
sparse-dense-interleaved        : 2.27206e-05 ± 0.666238 %
dense-sparse-interleaved        : 2.24958e-05 ± 0.518085 %
dense-sparse-densified          : 1.34508e-05 ± 2.12782 %
dense-sparse-densified2         : 1.33309e-05 ± 2.42168 %
dense-sparse-unstable           : 2.21002e-06 ± 3.7291 %
sparse-dense-unstable-unsorted  : 2.48519e-06 ± 3.48691 %
sparse-sparse-interleaved       : 1.77458e-05 ± 1.01638 %

$ ./build/basic -d 0.2 -l 100000
dense-dense                     : 9.64672e-05 ± 0.789742 %
sparse-dense-interleaved        : 0.000221988 ± 0.745541 %
dense-sparse-interleaved        : 0.000218575 ± 0.375344 %
dense-sparse-densified          : 0.000143562 ± 1.76462 %
dense-sparse-densified2         : 0.000144681 ± 2.35821 %
dense-sparse-unstable           : 2.26639e-05 ± 5.27362 %
sparse-dense-unstable-unsorted  : 3.46126e-05 ± 6.63714 %
sparse-sparse-interleaved       : 0.000175972 ± 1.78535 %

$ ./build/basic -d 0.05 -l 10000
dense-dense                     : 9.88294e-06 ± 1.88603 %
sparse-dense-interleaved        : 1.01129e-05 ± 0.315345 %
dense-sparse-interleaved        : 1.04033e-05 ± 1.98946 %
dense-sparse-densified          : 1.25296e-05 ± 1.44665 %
dense-sparse-densified2         : 1.12924e-05 ± 1.6 %
dense-sparse-unstable           : 7.4941e-07 ± 8.33484 %
sparse-dense-unstable-unsorted  : 7.6401e-07 ± 4.62903 %
sparse-sparse-interleaved       : 4.40413e-06 ± 0.49678 %

$ ./build/basic -d 0.5 -l 10000
dense-dense                     : 9.68416e-06 ± 0.286376 %
sparse-dense-interleaved        : 3.6256e-05 ± 0.249358 %
dense-sparse-interleaved        : 3.66027e-05 ± 0.144778 %
dense-sparse-densified          : 1.473e-05 ± 1.4579 %
dense-sparse-densified2         : 1.57786e-05 ± 1.48865 %
dense-sparse-unstable           : 5.17306e-06 ± 2.44265 %
sparse-dense-unstable-unsorted  : 5.96087e-06 ± 3.14494 %
sparse-sparse-interleaved       : 4.27666e-05 ± 0.212157 %
```

For fine-tuning, we also consider `sparse-dense-unstable` where the query is sparse and the reference is dense.
This will not be efficient as `dense-sparse-unstable` as we still need to compute the scaled ranks for the dense reference.
(Note that, in the basic case, this is the same as `dense-sparse-unstable` as the dense scaled ranks are computed ahead of time.)

```bash
$ ./build/fine_tune -d 0.2 -l 10000
dense-dense                     : 2.74382e-05 ± 1.20837 %
sparse-dense-interleaved        : 4.11986e-05 ± 0.543336 %
dense-sparse-interleaved        : 0.000105643 ± 0.283504 %
dense-sparse-densified          : 2.34307e-05 ± 1.17383 %
dense-sparse-densified2         : 2.35664e-05 ± 1.00415 %
dense-sparse-unstable           : 1.12498e-05 ± 0.532998 %
sparse-dense-unstable           : 1.89637e-05 ± 1.10841 %
sparse-dense-unstable-unsorted  : 1.94959e-05 ± 2.36502 %
sparse-sparse-interleaved       : 0.000100596 ± 0.716732 %

$ ./build/fine_tune -d 0.2 -l 100000
dense-dense                     : 0.000310617 ± 2.85776 %
sparse-dense-interleaved        : 0.000447954 ± 1.33866 %
dense-sparse-interleaved        : 0.00128377 ± 0.232774 %
dense-sparse-densified          : 0.000283887 ± 0.925263 %
dense-sparse-densified2         : 0.000302819 ± 1.34167 %
dense-sparse-unstable           : 0.000140279 ± 2.07608 %
sparse-dense-unstable           : 0.000235726 ± 3.54821 %
sparse-dense-unstable-unsorted  : 0.00024886 ± 2.73481 %
sparse-sparse-interleaved       : 0.00122911 ± 0.325709 %

$ ./build/fine_tune -d 0.05 -l 10000
dense-dense                     : 2.20143e-05 ± 0.869893 %
sparse-dense-interleaved        : 2.35321e-05 ± 1.5116 %
dense-sparse-interleaved        : 2.72197e-05 ± 0.817433 %
dense-sparse-densified          : 1.56123e-05 ± 2.28693 %
dense-sparse-densified2         : 1.35633e-05 ± 1.50595 %
dense-sparse-unstable           : 2.94963e-06 ± 1.136 %
sparse-dense-unstable           : 1.23217e-05 ± 2.4734 %
sparse-dense-unstable-unsorted  : 1.229e-05 ± 1.81857 %
sparse-sparse-interleaved       : 2.10924e-05 ± 1.07797 %

$ ./build/fine_tune -d 0.5 -l 10000
dense-dense                     : 3.74883e-05 ± 2.30634 %
sparse-dense-interleaved        : 6.52441e-05 ± 1.31196 %
dense-sparse-interleaved        : 0.000267795 ± 0.410695 %
dense-sparse-densified          : 4.03746e-05 ± 2.54741 %
dense-sparse-densified2         : 4.40882e-05 ± 2.06429 %
dense-sparse-unstable           : 2.77462e-05 ± 1.05438 %
sparse-dense-unstable           : 3.26929e-05 ± 2.27298 %
sparse-dense-unstable-unsorted  : 3.31167e-05 ± 2.45959 %
sparse-sparse-interleaved       : 0.00027094 ± 0.368072 %
```

Some comments:

- The interleaving approaches perform poorly.
  I would guess that the interleaving involves so many branch mispredictions that it's just faster to do more memory accesses and create a dense array. 
- `dense-sparse-unstable` is consistently good and indicates that storing sparse references is beneficial.
  I would wager that some numerical inaccuracy is acceptable here as distances close to zero would result in correlations close to 1, at which point small errors can be ignored.
  It is probably a good idea to clamp the correlations to [-1, 1] to avoid egregiously wrong results, e.g., correlations > 1.
- We should sort the query to improve cache locality, even though the unstable sparse/dense calculations don't strictly need sorting.
  It's quite a bit faster for the basic L2 calculations, it's no worse for the fine-tuning calculations,
  and the cost of sorting each query is amortized over the many L2 calculations involving that query.
