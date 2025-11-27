// RacEr Kernel for 4x4 Matrix Multiplication

#define RacEr_global_X 4
#define RacEr_global_Y 4
#define RacEr_tiles_X 3
#define RacEr_tiles_Y 3

#include "RacEr_manycore.h"
#include "RacEr_set_tile_x_y.h"

#define RacEr_TILE_GROUP_X_DIM RacEr_tiles_X
#define RacEr_TILE_GROUP_Y_DIM RacEr_tiles_Y

#include "RacEr_tile_group_barrier.h"

// Initialize tile group barrier
INIT_TILE_GROUP_BARRIER(r_barrier, c_barrier, 0, RacEr_tiles_X - 1, 0, RacEr_tiles_Y - 1);

int __attribute__((noinline)) kernel_matrix_mult_4x4(
    float *A, float *B, float *C, int N, int block_size)
{
  // Calculate starting index based on tile group position
  int start_idx = block_size *
                  (__RacEr_tile_group_id_y * __RacEr_grid_dim_x + __RacEr_tile_group_id_x);

  // Each tile works on a subset of output elements
  for (int elem_idx = __RacEr_id; elem_idx < block_size; elem_idx += RacEr_tiles_X * RacEr_tiles_Y)
  {
    int global_idx = start_idx + elem_idx; // linear index

    int i = global_idx / N;  // Row index
    int j = global_idx % N;  // Column index

    // Compute dot product of row i of A with column j of B
    float sum = 0.0f;
    for (int k = 0; k < N; k++) {
      sum += A[i * N + k] * B[k * N + j];
    }

    C[global_idx] = sum;
  }

  // Synchronize all tiles in the group
  RacEr_tile_group_barrier(&r_barrier, &c_barrier);

  return 0;
}
