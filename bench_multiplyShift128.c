/*****************************************************************
 * BIKE-style multiply-shift sampler for HQC (constant-time, 32-bit)
 *
 * Given a uniform 32-bit a, and divisor n = PARAM_N - i,
 *    q = ⌊(a·n) / 2³²⌋        // high 32 bits of the 64-bit product
 * return q                  // in [0, n)
 *****************************************************************/
 #include <stdio.h>
 #include <stdint.h>
 #include <x86intrin.h>   // for __rdtscp()
 #include <assert.h>
 #include <stdlib.h>
 
 #define PARAM_N 17669u   /* HQC-128 */
 
 static inline uint32_t reduce_mulshift(uint32_t a, size_t i) {
     uint32_t n    = PARAM_N - (uint32_t)i;   // current modulus
     uint64_t prod = (uint64_t)a * n;         // 32×32→64
     return prod >> 32;                       // floor(a·n/2^32) in [0,n)
 }
 
 /* quick self-test on edge cases */
 static void validate_edge_cases(void) {
     for (size_t i = 0; i < 75; i++) {
         uint32_t n = PARAM_N - (uint32_t)i;
         // small a
         assert(reduce_mulshift(0,     i) == 0);
         assert(reduce_mulshift(n,     i) < n);
         assert(reduce_mulshift(n + 1, i) < n);
         // large a
         assert(reduce_mulshift(UINT32_MAX, i) == n - 1);
     }
     printf("edge-case tests OK\n");
 }
 
 int main(void) {
     const size_t ITERS = 10000000;
     const size_t RUNS  = 10;
 
     // pre-generate 32-bit “random” inputs
     uint32_t *inp = aligned_alloc(64, ITERS * sizeof(uint32_t));
     for (size_t i = 0; i < ITERS; ++i) {
         inp[i] = 0x9E3779B9u * (uint32_t)i + 0x7F4A7C15u;
     }
 
     validate_edge_cases();
 
     // benchmark
     uint64_t tot = 0, sum = 0;
     for (size_t run = 0; run < RUNS; ++run) {
         unsigned junk;
         uint64_t st = __rdtscp(&junk);
         for (size_t i = 0; i < ITERS; ++i) {
             sum += reduce_mulshift(inp[i], i % 75);
         }
         uint64_t en = __rdtscp(&junk);
         tot += en - st;
     }
 
     printf("checksum          %llu\n", (unsigned long long)sum);
     printf("avg cycles/call   %.3f\n",
            (double)tot / (RUNS * ITERS));
 
     free(inp);
     return 0;
 }
 