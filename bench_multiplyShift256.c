/*****************************************************************
 * BIKE-style multiply-shift sampler for HQC-256 (constant-time, 32-bit)
 *
 * Given a uniform 32-bit a and divisor n = PARAM_N − i
 *     q = ⌊(a·n)/2³²⌋              // high 32 bits of the 64-bit product
 * return q   (in [0,n) )
 *****************************************************************/
 #include <stdio.h>
 #include <stdint.h>
 #include <x86intrin.h>   /* __rdtscp */
 #include <assert.h>
 #include <stdlib.h>
 
 #define PARAM_N  57637u      /* HQC-256 */
 #define OMEGA    149u        /* number of different moduli */
 
 static inline uint32_t reduce_mulshift(uint32_t a, size_t i)
 {
     uint32_t n    = PARAM_N - (uint32_t)i;   /* current modulus          */
     uint64_t prod = (uint64_t)a * n;         /* 32×32 → 64 product       */
     return prod >> 32;                       /* ⌊a·n / 2³²⌋  (∈[0,n) )  */
 }
 
 /* ---------- quick edge-case self-test --------------------------- */
 static void validate_edge_cases(void)
 {
     for (size_t i = 0; i < OMEGA; ++i) {
         uint32_t n = PARAM_N - (uint32_t)i;
 
         /* small a */
         assert(reduce_mulshift(0,     i) == 0);
         assert(reduce_mulshift(n,     i) <  n);
         assert(reduce_mulshift(n + 1, i) <  n);
 
         /* largest a */
         assert(reduce_mulshift(UINT32_MAX, i) == n - 1);
     }
     puts("edge-case tests OK");
 }
 
 /* -------------------------- benchmark --------------------------- */
 int main(void)
 {
     const size_t ITERS = 10000000;
     const size_t RUNS  = 10;
 
     /* deterministic “random” inputs */
     uint32_t *inp = aligned_alloc(64, ITERS * sizeof(uint32_t));
     for (size_t k = 0; k < ITERS; ++k)
         inp[k] = 0x9E3779B9u * k + 0x7F4A7C15u;
 
     validate_edge_cases();
 
     uint64_t tot_cycles = 0, checksum = 0;
     for (size_t r = 0; r < RUNS; ++r) {
         unsigned junk;
         uint64_t t0 = __rdtscp(&junk);
         for (size_t j = 0; j < ITERS; ++j)
             checksum += reduce_mulshift(inp[j], j % OMEGA);
         uint64_t t1 = __rdtscp(&junk);
         tot_cycles += t1 - t0;
     }
 
     printf("checksum          %llu\n", (unsigned long long)checksum);
     printf("avg cycles/call   %.3f\n",
            (double)tot_cycles / (RUNS * ITERS));
 
     free(inp);
     return 0;
 }
 