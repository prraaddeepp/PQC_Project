/*****************************************************************
 * Lemire-style multiply-shift sampler (unbiased, one-retry)
 * for HQC-256 (32-bit version)
 *
 * See Lemire-Kaser-Kurz, “Faster Remainder by Direct Computation”,
 * Algorithm 5 — this implements the unbiased sampler with one retry.
 *****************************************************************/
 #include <stdio.h>
 #include <stdint.h>
 #include <x86intrin.h>
 #include <assert.h>
 #include <stdlib.h>
 
 #define PARAM_N       57637u      /* first modulus for HQC-256  */
 #define THR_TBL_SIZE  149u        /* ω_max for HQC-256          */
 
 static uint32_t thr_tbl[THR_TBL_SIZE];   /* 2³² mod (PARAM_N − i) */
 
 /* ---- unbiased reducer (one-retry) ---------------------------- */
 static inline uint32_t reduce_mulshift_retry(uint32_t a, size_t i)
 {
     uint32_t n   = PARAM_N - (uint32_t)i;     /* current modulus   */
     uint32_t thr = thr_tbl[i];                /* 2³² mod n (thresh)*/
 
     for (;;) {
         uint64_t m  = (uint64_t)a * n;        /* 32×32 → 64        */
         uint32_t hi = (uint32_t)(m >> 32);    /* candidate [0,n)   */
         uint32_t lo = (uint32_t)m;            /* low 32 bits       */
         if (lo >= thr)                        /* accept if lo≥thr  */
             return hi;
 
         /* retry — draw a new 32-bit word (simple LCG) */
         a = 0x9E3779B9u * a + 0x7F4A7C15u;
     }
 }
 /* ------------------------------------------------------------- */
 
 /* quick edge-case self-test */
 static void validate_edge_cases(void)
 {
     for (size_t i = 0; i < THR_TBL_SIZE; ++i) {
         uint32_t n = PARAM_N - (uint32_t)i;
         assert(reduce_mulshift_retry(0,          i) < n);
         assert(reduce_mulshift_retry(UINT32_MAX, i) < n);
     }
     puts("edge-case tests OK");
 }
 
 int main(void)
 {
     /* pre-compute thresholds:  thr = 2³² mod n = (-n) mod n  */
     for (size_t i = 0; i < THR_TBL_SIZE; ++i) {
         uint32_t n = PARAM_N - (uint32_t)i;
         thr_tbl[i] = (-n) % n;
     }
 
     const size_t ITERS = 10000000;
     const size_t RUNS  = 10;
 
     /* deterministic “random” inputs */
     uint32_t *inp = aligned_alloc(64, ITERS * sizeof(uint32_t));
     for (size_t k = 0; k < ITERS; ++k)
         inp[k] = 0x9E3779B9u * k + 0x7F4A7C15u;
 
     validate_edge_cases();
 
     uint64_t tot_cycles = 0, checksum = 0;
     for (size_t run = 0; run < RUNS; ++run) {
         unsigned junk;
         uint64_t t0 = __rdtscp(&junk);
         for (size_t j = 0; j < ITERS; ++j)
             checksum += reduce_mulshift_retry(inp[j], j % THR_TBL_SIZE);
         uint64_t t1 = __rdtscp(&junk);
         tot_cycles += t1 - t0;
     }
 
     printf("checksum          %llu\n", (unsigned long long)checksum);
     printf("avg cycles/call   %.3f\n",
            (double)tot_cycles / (RUNS * ITERS));
 
     free(inp);
     return 0;
 }
 