#ifndef MATRIX_H
#define MATRIX_H

#include <stdlib.h>

typedef struct {
  float *data;
  size_t rows;
  size_t cols;
} Matrix;

#define matrix_at(matrix, row, col) matrix.data[row * matrix.cols + col]
#define matrix_at_transposed(matrix, row, col)                                 \
  matrix.data[col * matrix.cols + row]

// Allocation included
Matrix init_matrix(size_t rows, size_t cols);
void free_matrix(Matrix m);
// Allocation included
Matrix matrix_transpose(Matrix m);

void matrix_add(Matrix out, Matrix a, Matrix b); // TODO: Implement
void matrix_add_inplace(Matrix out, Matrix b);
void matrix_sub(Matrix out, Matrix a, Matrix b); // TODO: Implement
void matrix_sub_inplace(Matrix out, Matrix b);

void matrix_mul_scalar(Matrix out, Matrix m, float s); // TODO: Implement
void matrix_mul_scalar_inplace(Matrix out, float s);

void matrix_mul(Matrix out, Matrix a, Matrix b);
void matrix_mul_transposed_b(Matrix out, Matrix a, Matrix bT);
void matrix_mul_transpose_b(Matrix out, Matrix a, Matrix b);

#endif // MATRIX_H_

#ifdef MATRIX_IMPLEMENTATION
#ifndef MATRIX_IMPLEMENTATION_ONCE
#define MATRIX_IMPLEMENTATION_ONCE

#include <assert.h>
#include <string.h>

Matrix init_matrix(size_t rows, size_t cols) {
  Matrix m = {
      .data = (float *)calloc(rows * cols, sizeof(float)),
      .rows = rows,
      .cols = cols,
  };
  return m;
}

void free_matrix(Matrix m) {
  assert(m.data != NULL);
  free(m.data);
}

Matrix matrix_transpose(Matrix m) {
  Matrix out = init_matrix(m.cols, m.rows);
  for (size_t r = 0; r < m.rows; r++) {
    for (size_t c = 0; c < m.cols; c++) {
      out.data[c * out.cols + r] = m.data[r * m.cols + c];
    }
  }
  return out;
}

void matrix_add_inplace(Matrix out, Matrix b) {
  assert(out.rows == b.rows);
  assert(out.cols == b.cols);

  size_t size = out.rows * out.cols;
  for (size_t i = 0; i < size; ++i)
    out.data[i] += b.data[i];
}

void matrix_sub_inplace(Matrix out, Matrix b) {
  assert(out.rows == b.rows);
  assert(out.cols == b.cols);

  size_t size = out.rows * out.cols;
  for (size_t i = 0; i < size; ++i)
    out.data[i] -= b.data[i];
}

void matrix_mul_scalar_inplace(Matrix out, float s) {
  size_t size = out.rows * out.cols;
  for (size_t i = 0; i < size; ++i)
    out.data[i] *= s;
}

// General matrix-matrix multiplication
void matrix_mul(Matrix out, Matrix a, Matrix b) {
  // a: [M x K], b: [K x N] -> out: [M x N]
  assert(a.cols == b.rows);
  assert(out.rows == a.rows);
  assert(out.cols == b.cols);

  memset(out.data, 0, out.rows * out.cols * sizeof(float));

  for (size_t i = 0; i < a.rows; ++i) {
    for (size_t k = 0; k < a.cols; ++k) {
      float temp_a = matrix_at(a, i, k);

      for (size_t j = 0; j < b.cols; ++j) {
        matrix_at(out, i, j) += temp_a * matrix_at(b, k, j);
      }
    }
  }
}

void matrix_mul_transposed_b(Matrix out, Matrix a, Matrix bT) {
  // a: [M x K], bT: [N x K] -> out: [M x N]
  assert(a.cols == bT.cols);
  assert(out.rows == a.rows);
  assert(out.cols == bT.rows);

  memset(out.data, 0, out.rows * out.cols * sizeof(float));

  for (size_t i = 0; i < a.rows; ++i) {
    for (size_t j = 0; j < bT.rows; ++j) {
      float sum = 0;
      for (size_t k = 0; k < a.cols; ++k) {
        sum += matrix_at(a, i, k) * matrix_at_transposed(bT, k, j);
      }
      matrix_at(out, i, j) = sum;
    }
  }
}

// Since this function includes allocating and freeing a transposed matrix, it
// would not improve the performance a lot when this function is called very
// frequently. In that case you should usr `matrix_mul_transposed_b`
void matrix_mul_transpose_b(Matrix out, Matrix a, Matrix b) {
  // a: [M x K], b: [K x N] -> out: [M x N]
  assert(a.cols == b.rows);
  assert(out.rows == a.rows);
  assert(out.cols == b.cols);

  Matrix bT = matrix_transpose(b);
  matrix_mul_transposed_b(out, a, bT);
  free_matrix(bT);
}

#endif // MATRIX_IMPLEMENTATION_ONCE
#endif // MATRIX_IMPLEMENTATION
