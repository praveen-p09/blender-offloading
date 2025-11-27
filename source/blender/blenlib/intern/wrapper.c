// test wrapper with I/O and timing

#define _POSIX_C_SOURCE 199309L

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

extern int RacEr_matrix_mult_4x4(const float *A, const float *B, float *C);

static double ms_diff(struct timespec s, struct timespec e)
{
  double sec = (double)(e.tv_sec - s.tv_sec);
  double nsec = (double)(e.tv_nsec - s.tv_nsec);
  return sec * 1000.0 + nsec / 1.0e6;
}

void print_matrix(const char *name, const float *mat)
{
  printf("%s =\n", name);
  for (int i = 0; i < 4; i++) {
    printf("  ");
    for (int j = 0; j < 4; j++) {
      printf("%8.3f ", mat[i * 4 + j]);
    }
    printf("\n");
  }
}

int main(int argc, char *argv[])
{
  const int MATRIX_SIZE = 16;

  printf("RacEr 4x4 Matrix Multiplication Test\n");
  printf("=====================================\n\n");

  float *A = (float *)malloc(MATRIX_SIZE * sizeof(float));
  float *B = (float *)malloc(MATRIX_SIZE * sizeof(float));
  float *C = (float *)malloc(MATRIX_SIZE * sizeof(float));

  if (!A || !B || !C) {
    fprintf(stderr, "malloc failed\n");
    return 1;
  }

  printf("Enter 16 values for matrix A (row by row):\n");
  for (int i = 0; i < MATRIX_SIZE; i++) {
    if (scanf("%f", &A[i]) != 1) {
      fprintf(stderr, "Failed reading A[%d]\n", i);
      return 1;
    }
  }

  printf("\nEnter 16 values for matrix B (row by row):\n");
  for (int i = 0; i < MATRIX_SIZE; i++) {
    if (scanf("%f", &B[i]) != 1) {
      fprintf(stderr, "Failed reading B[%d]\n", i);
      return 1;
    }
  }

  printf("\n");
  print_matrix("Matrix A", A);
  printf("\n");
  print_matrix("Matrix B", B);
  printf("\n");

  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);

  int rc = RacEr_matrix_mult_4x4(A, B, C);

  clock_gettime(CLOCK_MONOTONIC, &t1);

  if (rc != 0) {
    fprintf(stderr, "RacEr_matrix_mult_4x4 failed (rc=%d)\n", rc);
    return 1;
  }

  double ms = ms_diff(t0, t1);

  print_matrix("Result Matrix C = A * B", C);
  printf("\n");

  printf("Performance Metrics:\n");
  printf("-------------------\n");
  printf("Execution_Time_ms = %.6f\n", ms);
  printf("Total_FLOPs = 128\n");
  printf("Throughput_GFLOPs_per_sec = %.6f\n", (ms > 0.0) ? (128.0 / (ms * 1.0e6)) : 0.0);

  free(A);
  free(B);
  free(C);

  return 0;
}
