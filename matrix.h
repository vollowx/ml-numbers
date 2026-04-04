#ifndef MATRIX_H_
#define MATRIX_H_

#include <assert.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  float *data;
  size_t rows;
  size_t cols;
} Matrix;

Matrix init_matrix(size_t rows, size_t cols);
void free_matrix(Matrix m);
Matrix matrix_transpose(Matrix m);
void matrix_plus(Matrix out, Matrix a, Matrix b);
void matrix_plus_inplace(Matrix a, Matrix b);
void matrix_mul(Matrix out, Matrix a, Matrix b);
void matrix_mul_transposed_b(Matrix out, Matrix a, Matrix bT);

#endif // MATRIX_H_

#ifdef MATRIX_IMPLEMENTATION

Matrix init_matrix(size_t rows, size_t cols) {
  Matrix m = {
      .data = (float *)calloc(rows * cols, sizeof(float)),
      .rows = rows,
      .cols = cols,
  };
  return m;
}

void free_matrix(Matrix m) {
  if (m.data != NULL) {
    free(m.data);
  }
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

void matrix_plus_inplace(Matrix a, Matrix b) {
  assert(a.rows == b.rows);
  assert(a.cols == b.cols);

  size_t size = a.rows * a.cols;
  for (size_t i = 0; i < size; ++i)
    a.data[i] += b.data[i];
}

void matrix_mul(Matrix out, Matrix a, Matrix b) {
  assert(a.cols == b.rows);
  assert(out.rows == a.rows);
  assert(out.cols == b.cols);

  memset(out.data, 0, out.rows * out.cols * sizeof(float));

  for (int i = 0; i < a.rows; ++i) {
    for (int k = 0; k < a.cols; ++k) {
      float temp_a = a.data[i * a.cols + k];

      for (int j = 0; j < b.cols; ++j) {
        out.data[i * out.cols + j] += temp_a * b.data[k * b.cols + j];
      }
    }
  }
}

void matrix_mul_transposed_b(Matrix out, Matrix a, Matrix bT) {
  // Dimensions: a is [M x K], bT is [N x K] -> Result is [M x N]
  assert(a.cols == bT.cols);

  memset(out.data, 0, out.rows * out.cols * sizeof(float));

  for (int i = 0; i < a.rows; ++i) {
    for (int j = 0; j < bT.rows; ++j) { // bT.rows is the original b.cols
      float sum = 0;
      for (int k = 0; k < a.cols; ++k) {
        sum += a.data[i * a.cols + k] * bT.data[j * bT.cols + k];
      }
      out.data[i * out.cols + j] = sum;
    }
  }
}

#endif // MATRIX_IMPLEMENTATION
