// Bridge file for offloading matrix operations to RacEr FPGA accelerator

#include "BLI_math_matrix.hh"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <cstring>
#include <iostream>

namespace blender {

// Implemented in host.c
#ifdef WITH_RACER
extern "C" int RacEr_matrix_mult_4x4(const float *A_host, const float *B_host, float *C_host);
#endif

// Called from math_matrix.cc operator*
void offload_matrix_multiply(const float4x4 &a, const float4x4 &b, float4x4 &result)
{
  // Cast Blender's float4x4 to flat float arrays, as float[4][4] - row-major and contiguous
#ifdef WITH_RACER
  // Try hardware acceleration
  const float *A_ptr = reinterpret_cast<const float *>(&a[0][0]);
  const float *B_ptr = reinterpret_cast<const float *>(&b[0][0]);
  float *C_ptr = reinterpret_cast<float *>(&result[0][0]);

  int rc = RacEr_matrix_mult_4x4(A_ptr, B_ptr, C_ptr);

  if (rc == 0) {
    // Success!
    return;
  }

  // If we get here, hardware failed - fall through to CPU
  std::cerr << "[RacEr Bridge] Hardware offload failed (rc=" << rc << "), using CPU fallback\n";
#endif

    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
        result[i][j] = 0.0f;
        for (int k = 0; k < 4; k++) {
          result[i][j] += a[i][k] * b[k][j];
        }
      }
    }
  }
}

// Matrix inversion using Eigen (CPU-based), for future hardware acceleration
//float4x4 offload_matrix_invert(const float4x4 &input)
//{
//  Eigen::Map<const Eigen::Matrix4f> mat(&input[0][0]);
//  Eigen::Matrix4f inv = mat.inverse();
//
//  float4x4 result;
//  std::memcpy(&result[0][0], inv.data(), sizeof(float) * 16);
//
//  return result;
//}
//
//// Matrix transpose (CPU-based for now)
//float4x4 offload_matrix_transpose(const float4x4 &input)
//{
//  float4x4 result;
//  for (int i = 0; i < 4; i++) {
//    for (int j = 0; j < 4; j++) {
//      result[i][j] = input[j][i];
//    }
//  }
//  return result;
//}

}  // namespace blender
