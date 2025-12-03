#########################################################
# Network Configutaion
# If not configured, Will use default Values
        RacEr_global_X ?= $(RacEr_tiles_X)
        RacEr_global_Y ?= $(RacEr_tiles_Y)+1

#########################################################
#Tile group configuration
# If not configured, Will use default Values
        RacEr_tiles_org_X ?= 0
        RacEr_tiles_org_Y ?= 1

# If not configured, Will use default Values
        RacEr_tiles_X ?= 2
        RacEr_tiles_Y ?= 2


all: main.run


KERNEL_NAME ?=kernel_matrix_mul

OBJECT_FILES=main.o RacEr_set_tile_x_y.o RacEr_printf.o kernel_matrix_mul.o

include ../../Makefile.include


main.riscv: $(OBJECT_FILES) $(SPMD_COMMON_OBJECTS) ../../common/crt.o
        $(RISCV_LINK) $(OBJECT_FILES) $(SPMD_COMMON_OBJECTS) -o $@ $(RISCV_LINK_OPTS)


main.o: Makefile

include ../../../mk/Makefile.tail_rules