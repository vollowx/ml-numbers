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
void matrix_plus_inplace(Matrix a, Matrix b);
void matrix_plus(Matrix out, Matrix a, Matrix b);
void matrix_mul(Matrix out, Matrix a, Matrix b);

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
    for (int j = 0; j < a.cols; ++j) {
      float temp_a = a.data[i * a.cols + j];

      for (int k = 0; k < b.cols; ++k) {
        out.data[i * out.cols + k] += temp_a * b.data[j * b.cols + k];
      }
    }
  }
}

#endif // MATRIX_IMPLEMENTATION
