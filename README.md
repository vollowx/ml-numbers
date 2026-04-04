# ML Number Recognition

- https://en.wikipedia.org/wiki/Matrix_(mathematics)
- https://en.wikipedia.org/wiki/Dot_product

- [x] Use matrix transposing when calculating dot product
- [ ] Use Lisp to instruct

[Initial design](./number_recognition.c) does not use matrix and takes roughly
0.11s to finish [instructions](./instructions.txt) on my machine.

The [second one](./main.c) takes roughly 0.17s using matrix without transposing
and about 0.12s with transposing.

Interestingly, without the `-O3` flag, the transposed matrix version takes
0.64s, faster than the original one (0.7s).
