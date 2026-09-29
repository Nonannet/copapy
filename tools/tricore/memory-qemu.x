/*
 * Memory description for running the bare metal runner (coparun_baremetal.c)
 * on the qemu tricore_testboard, based on memory-tc27xx.x of the tricore-gcc
 * toolchain. BSS, stack and heap are placed in the 4 MiB external RAM,
 * the last MiB of it (0xa1300000) is reserved for the command stream.
 * The CSA stays in the internal data RAM, since it must be addressable
 * by the 16 bit offset of the context pointers.
 */
__USTACK_SIZE = 16K;
__ISTACK_SIZE = 256;
__HEAP_SIZE = 2M;
__CSA_SIZE = 16K;
__TRICORE_DERIVATE_MEMORY_MAP__ = 0x2700;
MEMORY
{
  PMU_PFLASH0 (rx!p): org = 0x80000000, len = 2M
  DMI_LDRAM (w!xp): org = 0xd0000000, len = 48K
  EXT_DRAM (w!xp): org = 0xa1000000, len = 3M
  PCP_PRAM (wp!x): org = 0, len = 0
  PCP_CMEM (rpx): org = 0, len = 0
}
REGION_ALIAS("DATA_MEM", DMI_LDRAM)
REGION_ALIAS("CODE_MEM", PMU_PFLASH0)
REGION_ALIAS("SDATA_MEM", DMI_LDRAM)
REGION_ALIAS("BSS_MEM", EXT_DRAM)
REGION_ALIAS("ZDATA_MEM", DMI_LDRAM)
REGION_ALIAS("CSA_MEM", DMI_LDRAM)
REGION_ALIAS("PCP_CODE", PCP_CMEM)
REGION_ALIAS("PCP_DATA", PCP_PRAM)
_. = ASSERT ((__TRICORE_DERIVATE_MEMORY_MAP__ == __TRICORE_DERIVATE_NAME__), "Using wrong Memory Map. This Map is for TC27XX");
INSERT BEFORE .startup
