# Sparse/dense L2 calculations

## Overview

[**singlepp**](https://github.com/SingleR-inc/singlepp) can construct reference objects from both dense and sparse matrices.
The reference objects store the values of each reference profile in sorted order, ready for calculation of scaled ranks during fine-tuning.
(For sparse matrices, we store the negative and positives values separately, omitting all of the zeros.)
Spearman's correlation is derived from the L2 norm of the difference between the vectors of scaled ranks of the query and reference profiles. 
The question is, what is the fastest way to compute this L2 norm?

## Algorithms

`dense-dense`: when both the query and reference are dense, we use the sorted reference profile vector to populate a dense array.
The L2 norm is then calculated by iterating over both arrays at once and summing the squared differences.

When the query is sparse and the reference is dense, we have several choices:

- The simplest is to just convert the query into a dense buffer, which is the same as the `dense-dense` calculation.
- `sparse-dense-interleaved`: another option is to interleave the summations. 
  Assuming that the query's non-zero values are sorted,
  we compute the sum of squared differences between the query array against the scaled rank of the reference's zero value,
  all the way up to the next non-zero reference value;

When the query is dense and the reference is sparse, we have several choices:

- `dense-sparse-interleaved`: same as `sparse-dense-interleaved`, but we need to resort the reference's non-zero values by index first. 
  This is because the reference's scaled rank vector would be originally sorted by value.
- `dense-sparse-densified`: we convert the reference's non-zero values into a dense array.
  This requires one memory access to fill the dense buffer with the scaled rank of the reference's zero value,
  and then another few accesses to populate all of the non-zero values.
  The summation then proceeds as described for the dense-dense case.
- `dense-sparse-densified2`: a variation of `dense-sparse-densified`.
  We populate the dense array with `x2 := x + zero_val` where `x` is the rank of each non-zero value and `zero_rank` is the scaled rank of the reference's zero value.
  Then the summation is performed on the squared value of `y - x2 - zero_val` across the array.
  We then reset the dense array by setting the values of the array to zero at the non-zero indices.
  This saves us a `memset` across the entire array at the cost of an extra summation during the summation loop.

When both the query and reference are sparse, we have several choices:

- `sparse-sparse-interleaved`: sort the reference's non-zero values by index,
  and then walk through the reference/query non-zero values by increasing index to perform the summation.
- `sparse-sparse-remapped`: we create a remapping vector of the query's non-zero values beforehand.
  For each reference profile, we iterate across its non-zero values and compute the sum of differences based on the remapping vector.
  We also set the remapping vector to some sentinel value if its non-zero value is used during the iteration over the reference's non-zero values. 
  We then iterate across the query's non-zero values to compute more sum of differences (for non-zero values that were not used during the reference iteration)
  and check how many sentinel values were set (which is used to determine any additional summation for the zero-valued ranks). 
- We could also just convert the query to a dense array beforehand, which boils down to any of the `dense-sparse-*` choices.

## Results

For an Intel i7-8850H running Ubuntu Linux, we get:

```bash
$ ./build/basic -d 0.2 -l 1000
dense-dense                     : 1.70422e-06 ± 1.05528 %
sparse-dense-interleaved        : 3.4572e-06 ± 8.44779 %
dense-sparse-interleaved        : 7.0104e-06 ± 1.12824 %
dense-sparse-densified          : 1.25697e-06 ± 1.33723 %
dense-sparse-densified2         : 1.258e-06 ± 1.08259 %
sparse-sparse-interleaved       : 7.25918e-06 ± 3.95215 %
sparse-sparse-remapped          : 1.16388e-06 ± 1.98426 %

$ ./build/basic -d 0.2 -l 10000
dense-dense                     : 2.58779e-05 ± 1.5919 %
sparse-dense-interleaved        : 3.81497e-05 ± 0.764505 %
dense-sparse-interleaved        : 9.62444e-05 ± 0.414227 %
dense-sparse-densified          : 1.48054e-05 ± 2.60711 %
dense-sparse-densified2         : 1.7071e-05 ± 2.61045 %
sparse-sparse-interleaved       : 9.52613e-05 ± 0.559252 %
sparse-sparse-remapped          : 1.1569e-05 ± 1.32694 %

$ ./build/basic -d 0.05 -l 1000
dense-dense                     : 1.61932e-06 ± 0.334882 %
sparse-dense-interleaved        : 1.66948e-06 ± 0.278109 %
dense-sparse-interleaved        : 2.11863e-06 ± 7.33209 %
dense-sparse-densified          : 1.41162e-06 ± 18.873 %
dense-sparse-densified2         : 1.02361e-06 ± 0.204857 %
sparse-sparse-interleaved       : 1.63816e-06 ± 9.54402 %
sparse-sparse-remapped          : 3.5426e-07 ± 46.3533 %

$ ./build/basic -d 0.5 -l 1000
dense-dense                     : 1.67456e-06 ± 0.484014 %
sparse-dense-interleaved        : 4.19953e-06 ± 0.301398 %
dense-sparse-interleaved        : 1.87793e-05 ± 2.57688 %
dense-sparse-densified          : 1.775e-06 ± 15.1344 %
dense-sparse-densified2         : 1.47832e-06 ± 0.562093 %
sparse-sparse-interleaved       : 1.95613e-05 ± 1.08883 %
sparse-sparse-remapped          : 4.86549e-06 ± 0.382964 %
```

Some comments:

- In general, the interleaving approaches perform poorly.
  I would guess that the interleaving involves so many branch mispredictions that it's just faster to do more memory accesses and create a dense array. 
- The choice between `dense-sparse-densified` and `dense-sparse-densified2` is largely a toss-up, I'd just pick the simpler one.
- At least there is some consistent benefit from preserving sparsity, as indicated by the good performance of `sparse-sparse-remapped` when the data is actually sparse.
  (Sparse data also benefits from reduced iterations and memory accesses when computing the scaled ranks themselves, as we can just skip all of the zeros.)
