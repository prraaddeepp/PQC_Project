/*****************************************************************
 * BIKE-style multiply-shift sampler for HQC-192 (constant-time)
 *
 * Given a uniform 32-bit a and divisor  n = PARAM_N – i,
 *      q = ⌊ (a · n) / 2³² ⌋           // high 32 bits of the 64-bit product
 * returns q in [0, n).
 *****************************************************************/
 #include <stdio.h>
 #include <stdint.h>
 #include <x86intrin.h>     // __rdtscp
 #include <assert.h>
 #include <stdlib.h>
 
 #define PARAM_N  35851u     /* HQC-192 */
 #define TABLE_SZ 114        /* ω_max for HQC-192 */
 
 static inline uint32_t reduce_mulshift(uint32_t a, size_t i)
 {
     uint32_t n    = PARAM_N - (uint32_t)i;      /* current modulus         */
     uint64_t prod = (uint64_t)a * n;            /* 32×32 → 64-bit product  */
     return (uint32_t)(prod >> 32);              /* ⌊a·n / 2³²⌋ ∈ [0,n)     */
 }
 
 /* ---------------- self-test on edge cases ---------------- */
 static void validate_edge_cases(void)
 {
     for (size_t i = 0; i < TABLE_SZ; ++i) {
         uint32_t n = PARAM_N - (uint32_t)i;
         /* small a */
         assert(reduce_mulshift(0,     i) == 0);
         assert(reduce_mulshift(n,     i) <  n);
         assert(reduce_mulshift(n + 1, i) <  n);
         /* large a */
         assert(reduce_mulshift(UINT32_MAX, i) == n - 1);
     }
     puts("edge-case tests OK");
 }
 
 int main(void)
 {
     const size_t ITERS = 10000000;
     const size_t RUNS  = 10;
 
     /* pre-generate deterministic “random” inputs */
     uint32_t *inp = aligned_alloc(64, ITERS * sizeof(uint32_t));
     for (size_t i = 0; i < ITERS; ++i)
         inp[i] = 0x9E3779B9u * (uint32_t)i + 0x7F4A7C15u;
 
     validate_edge_cases();
 
     /* ---------------- benchmark ---------------- */
     uint64_t tot_cycles = 0, sum = 0;
     for (size_t run = 0; run < RUNS; ++run) {
         unsigned junk;
         uint64_t st = __rdtscp(&junk);
         for (size_t i = 0; i < ITERS; ++i)
             sum += reduce_mulshift(inp[i], i % TABLE_SZ);
         uint64_t en = __rdtscp(&junk);
         tot_cycles += en - st;
     }
 
     printf("checksum          %llu\n", (unsigned long long)sum);
     printf("avg cycles/call   %.3f\n",
            (double)tot_cycles / (RUNS * ITERS));
 
     free(inp);
     return 0;
 }
 