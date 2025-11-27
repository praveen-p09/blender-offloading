// RacEr device management for 4x4 matrix multiplication

#define _XOPEN_SOURCE 700
#define _GNU_SOURCE

#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "RacEr_manycore.h"
#include "RacEr_cuda.h"
//#include "test_matrix_mult.h"

#define ALLOC_NAME "default_allocator"
#define KERNEL_PATH \
  "/home/ec2-user/SO_RacEr_Float/tfr/RacEr_bladerunner/" \
  "RacEr_manycore/software/spmd/RacEr_cuda_lite_runtime/" \
  "matrix_mult_4x4/main.riscv"

static int finish_device(RacEr_mc_device_t *dev)
{
  return RacEr_mc_device_finish(dev);
}

// macro that finishes device, returns an error code
#define FINISH_AND_RETURN(dev, code) \
  do { \
    (void)finish_device(&(dev)); \
    return (code); \
  } while (0)

// called from bridge.cpp, returns 0 on success, negative error code on failure
int RacEr_matrix_mult_4x4(const float *A_host, const float *B_host, float *C_host)
{
  const int N = 4;
  const int MATRIX_SIZE = 16;  // 4x4 

  if (!A_host || !B_host || !C_host) {
    errno = EINVAL;
    return -1;
  }

  const char *bin_path = KERNEL_PATH;
  if (!bin_path || !*bin_path)
    bin_path = BIN_PATH_DEFAULT;

  // Initialize device
  RacEr_mc_device_t device;
  int rc = RacEr_mc_device_init(&device, (char *)"t", 0);
  if (rc != HB_MC_SUCCESS)
    return -2;

  // Load kernel program
  rc = RacEr_mc_device_program_init(&device, (char *)bin_path, ALLOC_NAME, 0);
  if (rc != HB_MC_SUCCESS)
    FINISH_AND_RETURN(device, -3);

  // Allocate device memory for matrices 
  RacEr_mc_eva_t A_dev, B_dev, C_dev;
  size_t bytes = MATRIX_SIZE * sizeof(float);

  rc = RacEr_mc_device_malloc(&device, bytes, &A_dev);
  if (rc != HB_MC_SUCCESS)
    FINISH_AND_RETURN(device, -4);

  rc = RacEr_mc_device_malloc(&device, bytes, &B_dev);
  if (rc != HB_MC_SUCCESS)
    FINISH_AND_RETURN(device, -5);

  rc = RacEr_mc_device_malloc(&device, bytes, &C_dev);
  if (rc != HB_MC_SUCCESS)
    FINISH_AND_RETURN(device, -6);

  // Copy input matrices to device
  rc = RacEr_mc_device_memcpy(
      &device, (void *)(intptr_t)A_dev, (void *)A_host, bytes, HB_MC_MEMCPY_TO_DEVICE);
  if (rc != HB_MC_SUCCESS)
    FINISH_AND_RETURN(device, -7);

  rc = RacEr_mc_device_memcpy(
      &device, (void *)(intptr_t)B_dev, (void *)B_host, bytes, HB_MC_MEMCPY_TO_DEVICE);
  if (rc != HB_MC_SUCCESS)
    FINISH_AND_RETURN(device, -8);

  // Initialize output matrix to zero 
  rc = RacEr_mc_device_memset(&device, &C_dev, 0, bytes);
  if (rc != HB_MC_SUCCESS)
    FINISH_AND_RETURN(device, -9);

  // Configure tile group and grid dimensions 
  RacEr_mc_dimension_t tg_dim = (RacEr_mc_dimension_t){.x = 2, .y = 2};
  RacEr_mc_dimension_t grid_dim = (RacEr_mc_dimension_t){.x = 1, .y = 1};

  // Prepare kernel arguments
  uint32_t block_size = MATRIX_SIZE;
  uint32_t argv32[5] = {
      (uint32_t)A_dev, (uint32_t)B_dev, (uint32_t)C_dev, (uint32_t)N, block_size};

  // Enqueue kernel 
  rc = RacEr_mc_kernel_enqueue(
      &device, grid_dim, tg_dim, "kernel_matrix_mult_4x4", 5, (const uint32_t *)argv32);
  if (rc != HB_MC_SUCCESS)
    FINISH_AND_RETURN(device, -10);

  // Execute kernel 
  rc = RacEr_mc_device_tile_groups_execute(&device);
  if (rc != HB_MC_SUCCESS)
    FINISH_AND_RETURN(device, -11);

  // Copy result back to host 
  rc = RacEr_mc_device_memcpy(
      &device, (void *)C_host, (void *)(intptr_t)C_dev, bytes, HB_MC_MEMCPY_TO_HOST);
  if (rc != HB_MC_SUCCESS)
    FINISH_AND_RETURN(device, -12);

  // Clean up device
  rc = finish_device(&device);
  if (rc != HB_MC_SUCCESS)
    return -13;

  return 0;
}
