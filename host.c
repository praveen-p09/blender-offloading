// ============================================================
// host.c - Launches kernel_matrix_mul on RacEr manycore FPGA
// ============================================================

#define _XOPEN_SOURCE 700
#define _GNU_SOURCE

#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "RacEr_manycore_cuda.h"

#define ALLOC_NAME   "default_allocator"

// NOTE: make sure this file exists after compiling kernel:
// main.riscv must match kernel_matrix_mul signature exactly.
#define KERNEL_PATH \
"/home/ec2-user/RacEr_Float/tfr/RacEr_bladerunner/" \
"RacEr_manycore/software/spmd/RacEr_cuda_lite_runtime/" \
"matrix_mult_4x4/main.riscv"


// --------------------- Utility cleanup ----------------------

static int finish_device(RacEr_mc_device_t *dev)
{
    return RacEr_mc_device_finish(dev);
}

#define FINISH_AND_RETURN(dev, code) \
    do { finish_device(&(dev)); return (code); } while (0)


// ============================================================
//   Host function callable from C / C++ / Blender bridge
// ============================================================

int RacEr_matrix_mult_4x4(
        const float *A_host,
        const float *B_host,
        float       *C_host)
{
    const int M = 4, N = 4, P = 4;
    const int MATRIX_SIZE = 16;

    if (!A_host || !B_host || !C_host)
        return -1;

    RacEr_mc_device_t device;
    int rc;

    // -------------------- Initialize device --------------------
    rc = RacEr_mc_device_init(&device, (char*)"t", 0);
    if (rc != HB_MC_SUCCESS)
        return -2;

    rc = RacEr_mc_device_program_init(
            &device,
            (char*)KERNEL_PATH,
            ALLOC_NAME,
            0);

    if (rc != HB_MC_SUCCESS)
        FINISH_AND_RETURN(device, -3);


    // -------------------- Allocate device buffers --------------------
    size_t bytes = MATRIX_SIZE * sizeof(float);
    RacEr_mc_eva_t A_dev, B_dev, C_dev;

    if (RacEr_mc_device_malloc(&device, bytes, &A_dev) != HB_MC_SUCCESS)
        FINISH_AND_RETURN(device, -4);

    if (RacEr_mc_device_malloc(&device, bytes, &B_dev) != HB_MC_SUCCESS)
        FINISH_AND_RETURN(device, -5);

    if (RacEr_mc_device_malloc(&device, bytes, &C_dev) != HB_MC_SUCCESS)
        FINISH_AND_RETURN(device, -6);


    // -------------------- Copy A and B to device --------------------

    rc = RacEr_mc_device_memcpy(
            &device,
            (void*)(intptr_t)A_dev,
            (void*)A_host,
            bytes,
            HB_MC_MEMCPY_TO_DEVICE);

    if (rc != HB_MC_SUCCESS)
        FINISH_AND_RETURN(device, -7);

    rc = RacEr_mc_device_memcpy(
            &device,
            (void*)(intptr_t)B_dev,
            (void*)B_host,
            bytes,
            HB_MC_MEMCPY_TO_DEVICE);

    if (rc != HB_MC_SUCCESS)
        FINISH_AND_RETURN(device, -8);


    // -------------------- Zero output buffer --------------------

    rc = RacEr_mc_device_memset(&device, &C_dev, 0, bytes);
    if (rc != HB_MC_SUCCESS)
        FINISH_AND_RETURN(device, -9);



    int block_size_x = 4;
    int block_size_y = 4;

    RacEr_mc_dimension_t grid = { .x = 1, .y = 1 };

    RacEr_mc_dimension_t tg = { .x = 1, .y = 1 };

    uint32_t argv[8] = {
        (uint32_t)A_dev,
        (uint32_t)B_dev,
        (uint32_t)C_dev,
        (uint32_t)M,
        (uint32_t)N,
        (uint32_t)P,
        (uint32_t)block_size_y,
        (uint32_t)block_size_x
    };

    // ----------------------- Enqueue kernel -----------------------

    rc = RacEr_mc_kernel_enqueue(
            &device,
            grid,
            tg,
            "kernel_matrix_mul",   // MUST MATCH RISC-V symbol
            8,
            argv);

    if (rc != HB_MC_SUCCESS)
        FINISH_AND_RETURN(device, -10);


    // ----------------------- Run tile groups -----------------------

    rc = RacEr_mc_device_tile_groups_execute(&device);
    if (rc != HB_MC_SUCCESS)
        FINISH_AND_RETURN(device, -11);


    // ----------------------- Copy result back ----------------------

    rc = RacEr_mc_device_memcpy(
            &device,
            (void*)C_host,
            (void*)(intptr_t)C_dev,
            bytes,
            HB_MC_MEMCPY_TO_HOST);

    if (rc != HB_MC_SUCCESS)
        FINISH_AND_RETURN(device, -12);


    // ----------------------- Clean up -----------------------

    finish_device(&device);
    return 0;
}