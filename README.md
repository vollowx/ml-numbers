# ML Number Recognition

- https://en.wikipedia.org/wiki/Matrix_(mathematics)
- https://en.wikipedia.org/wiki/Dot_product

- [x] Use matrix transposing when calculating dot product
- [ ] Use Lisp to instruct

[Initial design](./nomatrix.c) does not use matrix and takes roughly
0.11s to finish [instructions](./instructions.txt) on my machine.

The [second one](./main.c) takes roughly 0.17s using matrix without transposing
and about 0.12s with transposing.

Interestingly, without the `-O3` flag, the transposed matrix version takes
0.64s, faster than the original one (0.7s).

Here is the average time cost with `-O3` optimizations:

```
$ hyperfine ./nomatrix
Benchmark 1: ./nomatrix
  Time (mean ± σ):      57.6 ms ±   0.3 ms    [User: 56.5 ms, System: 1.7 ms]
  Range (min … max):    57.1 ms …  58.7 ms    49 runs

$ hyperfine ./main
Benchmark 1: ./main
  Time (mean ± σ):      62.6 ms ±   0.3 ms    [User: 61.5 ms, System: 1.9 ms]
  Range (min … max):    62.2 ms …  63.4 ms    46 runs
```

and without `-O3` optimizations:

```
$ hyperfine ./nomatrix
Benchmark 1: ./nomatrix
  Time (mean ± σ):     550.1 ms ±   1.1 ms    [User: 548.4 ms, System: 2.6 ms]
  Range (min … max):   548.9 ms … 552.2 ms    10 runs

$ hyperfine ./main
Benchmark 1: ./main
  Time (mean ± σ):     532.5 ms ±   1.2 ms    [User: 531.2 ms, System: 1.5 ms]
  Range (min … max):   530.8 ms … 534.3 ms    10 runs
```

The time may vary since the durations are measured in different time.
