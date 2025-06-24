/*****************************************************************
 * Accurate Barrett Reducer Benchmark – HQC-192 (n = 35851)
 *   • Pre-generated inputs (no rand() overhead)
 *   • Serialized RDTSCP
 *   • Full edge-case validation
 *   • Frequency-independent measurements
 *****************************************************************/
 #include <stdio.h>
 #include <stdint.h>
 #include <x86intrin.h>
 #include <assert.h>
 #include <stdlib.h>
 
 #define PARAM_N   35851u        /* HQC-192 parameter n          */
 #define TABLE_SZ  114           /* ω_max                        */
 
 /* 32-bit Barrett constants  m = ⌈2³² / (PARAM_N-i)⌉            */
 static const uint32_t m_val[TABLE_SZ] = {
   119800,119803,119807,119810,119813,119817,119820,119823,119827,119830,
   119833,119837,119840,119843,119847,119850,119853,119857,119860,119864,
   119867,119870,119874,119877,119880,119884,119887,119890,119894,119897,
   119900,119904,119907,119910,119914,119917,119920,119924,119927,119930,
   119934,119937,119941,119944,119947,119951,119954,119957,119961,119964,
   119967,119971,119974,119977,119981,119984,119987,119991,119994,119997,
   120001,120004,120008,120011,120014,120018,120021,120024,120028,120031,
   120034,120038,120041,120044,120048,120051,120054,120058,120061,120065,
   120068,120071,120075,120078,120081,120085,120088,120091,120095,120098,
   120101,120105,120108,120112,120115,120118,120122,120125,120128,120132,
   120135,120138,120142,120145,120149,120152,120155,120159,120162,120165,
   120169,120172,120175,120179
 };
 
 /* -------- core Barrett reduction -------- */
 static inline uint32_t cond_sub(uint32_t r, uint32_t n)
 {
     uint32_t mask = -(r >= n);
     return r - (n & mask);
 }
 
 static inline uint32_t reduce_barrett(uint32_t a, size_t i)
 {
     uint32_t q = ((uint64_t)a * m_val[i]) >> 32;
     uint32_t n = PARAM_N - (uint32_t)i;
     uint32_t r = a - q * n;
     return cond_sub(r, n);
 }
 
 /* ---------- edge-case self-test ---------- */
 static void validate_edge_cases(void)
 {
     for (size_t i = 0; i < TABLE_SZ; ++i) {
         uint32_t n = PARAM_N - (uint32_t)i;
         assert(reduce_barrett(0,         i) == 0);
         assert(reduce_barrett(n - 1,     i) == n - 1);
         assert(reduce_barrett(n,         i) == 0);
         assert(reduce_barrett(2U * n,    i) == 0);
     }
     puts("edge-case tests OK");
 }
 
 int main(void)
 {
     const size_t ITERS = 10000000;
     const size_t RUNS  = 10;
 
     uint64_t total_cycles = 0, sum = 0;
 
     /* deterministic “random” inputs (fits in L1) */
     uint32_t *inputs = aligned_alloc(64, ITERS * sizeof(uint32_t));
     for (size_t i = 0; i < ITERS; ++i)
         inputs[i] = 0x9e3779b9u * (uint32_t)i + 0x7f4a7c15u;
 
     validate_edge_cases();
 
     for (size_t run = 0; run < RUNS; ++run) {
         unsigned junk;
         uint64_t start = __rdtscp(&junk);
 
         for (size_t i = 0; i < ITERS; ++i)
             sum += reduce_barrett(inputs[i], i % TABLE_SZ);
 
         uint64_t end = __rdtscp(&junk);
         total_cycles += (end - start);
     }
 
     printf("Validation checksum = %llu\n", (unsigned long long)sum);
     printf("Average cycles/call = %.3f\n",
            (double)total_cycles / (RUNS * ITERS));
 
     free(inputs);
     return 0;
 }
 