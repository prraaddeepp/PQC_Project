/*****************************************************************
 * Plain-division reducer (baseline) for HQC-256 moduli
 *     r = a % (PARAM_N - i)
 *
 * One variable-latency integer DIV/UDIV per call.
 *****************************************************************/
 #include <stdio.h>
 #include <stdint.h>
 #include <x86intrin.h>
 #include <assert.h>
 #include <stdlib.h>
 
 #define PARAM_N  57637u          /* HQC-256 first modulus      */
 #define OMEGA    149u            /* number of different moduli */
 
 /* single unspecialised %-reduction */
 static inline uint32_t reduce_idiv(uint32_t a, size_t i)
 {
     uint32_t n = PARAM_N - (uint32_t)i;
     return a % n;                /* compiler emits DIV/UDIV */
 }
 
 /* --------- sanity check on edge cases -------------------------- */
 static void validate_edge_cases(void)
 {
     for (size_t i = 0; i < OMEGA; ++i) {
         uint32_t n  = PARAM_N - (uint32_t)i;
         assert(reduce_idiv(0,     i) == 0);         /* r = 0             */
         assert(reduce_idiv(n - 1, i) == n - 1);     /* r = n-1           */
         assert(reduce_idiv(n,     i) == 0);         /* r = 0             */
     }
     puts("edge-case tests OK");
 }
 
 /* ----------------- benchmark harness --------------------------- */
 int main(void)
 {
     const size_t ITERS = 10000000;
     const size_t RUNS  = 10;
 
     /* deterministic “random” 32-bit inputs */
     uint32_t *inp = aligned_alloc(64, ITERS * sizeof(uint32_t));
     for (size_t k = 0; k < ITERS; ++k)
         inp[k] = 0x9E3779B9u * k + 0x7F4A7C15u;
 
     validate_edge_cases();
 
     uint64_t tot_cycles = 0, checksum = 0;
     for (size_t r = 0; r < RUNS; ++r) {
         unsigned junk;
         uint64_t t0 = __rdtscp(&junk);
         for (size_t j = 0; j < ITERS; ++j)
             checksum += reduce_idiv(inp[j], j % OMEGA);
         uint64_t t1 = __rdtscp(&junk);
         tot_cycles += t1 - t0;
     }
 
     printf("checksum          %llu\n", (unsigned long long)checksum);
     printf("avg cycles/call   %.3f\n",
            (double)tot_cycles / (RUNS * ITERS));
 
     free(inp);
     return 0;
 }
 