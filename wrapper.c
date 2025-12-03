// test wrapper with I/O, timing, and CPU vs RacEr comparison

#define _POSIX_C_SOURCE 199309L

#include <errno.h>
#include <math.h>
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

static void cpu_matmul_4x4(const float *A, const float *B, float *C)
{
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      float sum = 0.0f;
      for (int k = 0; k < 4; k++) {
        sum += A[i * 4 + k] * B[k * 4 + j];
      }
      C[i * 4 + j] = sum;
    }
  }
}

static void print_matrix(const char *name, const float *mat)
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

int main(void)
{
  const int MATRIX_SIZE = 16;

  printf("RacEr 4x4 Matrix Multiplication Test\n");
  printf("=====================================\n\n");

  float *A = (float *)malloc(MATRIX_SIZE * sizeof(float));
  float *B = (float *)malloc(MATRIX_SIZE * sizeof(float));
  float *C_fpga = (float *)malloc(MATRIX_SIZE * sizeof(float));
  float *C_cpu = (float *)malloc(MATRIX_SIZE * sizeof(float));

  if (!A || !B || !C_fpga || !C_cpu) {
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

  // CPU baseline
  struct timespec t0_cpu, t1_cpu;
  clock_gettime(CLOCK_MONOTONIC, &t0_cpu);
  cpu_matmul_4x4(A, B, C_cpu);
  clock_gettime(CLOCK_MONOTONIC, &t1_cpu);
  double ms_cpu = ms_diff(t0_cpu, t1_cpu);

  // RacEr (FPGA)
  struct timespec t0_fpga, t1_fpga;
  clock_gettime(CLOCK_MONOTONIC, &t0_fpga);

  int rc = RacEr_matrix_mult_4x4(A, B, C_fpga);

  clock_gettime(CLOCK_MONOTONIC, &t1_fpga);
  double ms_fpga = ms_diff(t0_fpga, t1_fpga);

  if (rc != 0) {
    fprintf(stderr, "RacEr_matrix_mult_4x4 failed (rc=%d)\n", rc);
    return 1;
  }

  print_matrix("CPU Result C_cpu", C_cpu);
  printf("\n");
  print_matrix("FPGA Result C_fpga", C_fpga);
  printf("\n");

  // Compare
  double max_abs_err = 0.0;
  for (int i = 0; i < MATRIX_SIZE; i++) {
    double diff = fabs((double)C_cpu[i] - (double)C_fpga[i]);
    if (diff > max_abs_err) {
      max_abs_err = diff;
    }
  }

  printf("Error Metrics:\n");
  printf("--------------\n");
  printf("Max_Abs_Error = %.9f\n", max_abs_err);
  printf("Status = %s\n\n", (max_abs_err < 1e-4) ? "PASS" : "FAIL");

  printf("Performance Metrics:\n");
  printf("-------------------\n");
  printf("CPU_Execution_Time_ms   = %.6f\n", ms_cpu);
  printf("FPGA_Execution_Time_ms  = %.6f\n", ms_fpga);
  printf("Total_FLOPs (4x4 mult)  = 128\n");
  printf("CPU_Throughput_GFLOPs   = %.6f\n", (ms_cpu > 0.0) ? (128.0 / (ms_cpu * 1.0e6)) : 0.0);
  printf("FPGA_Throughput_GFLOPs  = %.6f\n", (ms_fpga > 0.0) ? (128.0 / (ms_fpga * 1.0e6)) : 0.0);

  free(A);
  free(B);
  free(C_fpga);
  free(C_cpu);

  return 0;
}