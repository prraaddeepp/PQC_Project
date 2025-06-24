/*****************************************************************
 * Lemire-style multiply-shift sampler (unbiased, one-retry)
 * for HQC constant-time code, 32-bit version.
 *
 * Given  a ∈ [0,2³²) and divisor n = PARAM_N − i,
 *    m  = a·n                     // 64-bit product
 *    q  = hi32(m)                 // ⌊a·n / 2³²⌋  – candidate
 *    lo = lo32(m)
 * If lo < (2³² mod n)   =>  draw a new a  (“retry”)    [Alg. 5, lines 4-9]
 * Return q  (uniform in [0,n) )                         [Alg. 5, line 12]
 *****************************************************************/
 #include <stdio.h>
 #include <stdint.h>
 #include <x86intrin.h>
 #include <assert.h>
 #include <stdlib.h>
 
 #define PARAM_N 17669u          /* HQC-128 */
 #define THR_TBL_SIZE 75         /* ω_max */
 
 static uint32_t thr_tbl[THR_TBL_SIZE];   /* pre-computed 2³² mod (PARAM_N−i) */
 
 /* ---------- unbiased reducer (one-retry) -------------------- */
 static inline uint32_t reduce_mulshift_retry(uint32_t a, size_t i)
 {
     uint32_t n   = PARAM_N - (uint32_t)i;          /* current modulus         */
     uint32_t thr = thr_tbl[i];                     /* 2³² mod n  (threshold)  */
 
     for (;;) {
         uint64_t m  = (uint64_t)a * n;
         uint32_t hi = (uint32_t)(m >> 32);         /* candidate in [0,n)      */
         uint32_t lo = (uint32_t)m;                 /* lower 32 bits           */
         if (lo >= thr)                             /* Alg. 5: pass if lo≥thr  */
             return hi;
         /* retry — draw another 32-bit word (here a simple LCG) */
         a = 0x9E3779B9u * a + 0x7F4A7C15u;
     }
 }
 /* ------------------------------------------------------------- */
 
 /* quick self-test on edge cases */
 static void validate_edge_cases(void)
{
    for (size_t i = 0; i < THR_TBL_SIZE; ++i) {
        uint32_t n = PARAM_N - (uint32_t)i;
        /* The only guarantee is 0 ≤ r < n */
        assert(reduce_mulshift_retry(0,          i) < n);
        assert(reduce_mulshift_retry(UINT32_MAX, i) < n);
    }
    puts("edge-case tests OK");
}
 
 int main(void)
 {
     /* ---------- pre-compute thresholds -------------------- */
     for (size_t i = 0; i < THR_TBL_SIZE; ++i) {
         uint32_t n = PARAM_N - (uint32_t)i;
         thr_tbl[i] = (-n) % n;          /* = 2³² mod n  (Lemma 4.1) */
     }
 
     const size_t ITERS = 10000000;
     const size_t RUNS  = 10;
 
     /* deterministic “random” inputs */
     uint32_t *inp = aligned_alloc(64, ITERS * sizeof(uint32_t));
     for (size_t i = 0; i < ITERS; ++i)
         inp[i] = 0x9E3779B9u * i + 0x7F4A7C15u;
 
     validate_edge_cases();
 
     uint64_t tot = 0, sum = 0;
     for (size_t run = 0; run < RUNS; ++run) {
         unsigned junk;
         uint64_t st = __rdtscp(&junk);
         for (size_t i = 0; i < ITERS; ++i)
             sum += reduce_mulshift_retry(inp[i], i % THR_TBL_SIZE);
         uint64_t en = __rdtscp(&junk);
         tot += en - st;
     }
 
     printf("checksum          %llu\n", (unsigned long long)sum);
     printf("avg cycles/call   %.3f\n",
            (double)tot / (RUNS * ITERS));
 
     free(inp);
     return 0;
 }
 