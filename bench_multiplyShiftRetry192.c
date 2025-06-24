/*****************************************************************
 * Lemire-style multiply-shift sampler (unbiased, one-retry)
 * for HQC-192, 32-bit version.
 *
 *  Given a ∈ [0,2³²) and divisor n = PARAM_N − i,
 *     m  = a·n                     // 64-bit product
 *     q  = hi32(m)                 // ⌊a·n / 2³²⌋  – candidate
 *     lo = lo32(m)
 *  If lo < (2³² mod n)  → draw a new a  (“retry”)
 *  Return q   (uniform in [0,n) )
 *****************************************************************/
 #include <stdio.h>
 #include <stdint.h>
 #include <x86intrin.h>
 #include <assert.h>
 #include <stdlib.h>
 
 #define PARAM_N       35851u   /* -------- HQC-192 -------- */
 #define THR_TBL_SIZE  114      /* ω_max for HQC-192        */
 
 static uint32_t thr_tbl[THR_TBL_SIZE];   /* 2³² mod (PARAM_N − i) */
 
 /* -------- unbiased reduction with at most one retry -------- */
 static inline uint32_t reduce_mulshift_retry(uint32_t a, size_t i)
 {
     uint32_t n   = PARAM_N - (uint32_t)i;
     uint32_t thr = thr_tbl[i];              /* threshold */
 
     for (;;) {
         uint64_t m  = (uint64_t)a * n;      /* 32×32 → 64 */
         uint32_t hi = (uint32_t)(m >> 32);  /* candidate  */
         uint32_t lo = (uint32_t)m;          /* lower half */
 
         if (lo >= thr)                      /* pass test? */
             return hi;
 
         /* one retry – refresh a with a cheap LCG              */
         a = 0x9E3779B9u * a + 0x7F4A7C15u;
     }
 }
 
 /* ---------------- self-test on edge cases ------------------ */
 static void validate_edge_cases(void)
 {
     for (size_t i = 0; i < THR_TBL_SIZE; ++i) {
         uint32_t n = PARAM_N - (uint32_t)i;
         /* The reducer must always output < n                  */
         assert(reduce_mulshift_retry(0,          i) < n);
         assert(reduce_mulshift_retry(UINT32_MAX, i) < n);
     }
     puts("edge-case tests OK");
 }
 
 int main(void)
 {
     /* ---- pre-compute thresholds: 2³² mod (PARAM_N − i) ---- */
     for (size_t i = 0; i < THR_TBL_SIZE; ++i) {
         uint32_t n = PARAM_N - (uint32_t)i;
         thr_tbl[i] = (-n) % n;              /* equals 2³² mod n */
     }
 
     const size_t ITERS = 10000000;
     const size_t RUNS  = 10;
 
     /* deterministic “random” a-values */
     uint32_t *inp = aligned_alloc(64, ITERS * sizeof(uint32_t));
     for (size_t i = 0; i < ITERS; ++i)
         inp[i] = 0x9E3779B9u * i + 0x7F4A7C15u;
 
     validate_edge_cases();
 
     /* ----------------- benchmark loop ---------------------- */
     uint64_t tot_cycles = 0, sum = 0;
     for (size_t run = 0; run < RUNS; ++run) {
         unsigned junk;
         uint64_t st = __rdtscp(&junk);
         for (size_t i = 0; i < ITERS; ++i)
             sum += reduce_mulshift_retry(inp[i], i % THR_TBL_SIZE);
         uint64_t en = __rdtscp(&junk);
         tot_cycles += en - st;
     }
 
     printf("checksum          %llu\n", (unsigned long long)sum);
     printf("avg cycles/call   %.3f\n",
            (double)tot_cycles / (RUNS * ITERS));
 
     free(inp);
     return 0;
 }
 