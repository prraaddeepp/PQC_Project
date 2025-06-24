/*****************************************************************
 * Accurate Barrett-Reducer Benchmark  – HQC-256 (PARAM_N = 57637)
 *   – pre-generated inputs (no rand() in the hot loop)
 *   – serialized RDTSCP timing
 *   – full edge-case validation
 *****************************************************************/
 #include <stdio.h>
 #include <stdint.h>
 #include <x86intrin.h>
 #include <assert.h>
 #include <stdlib.h>
 
 #define PARAM_N   57637u           /* HQC-256 */
 #define TABLE_SZ  149u             /* ω_max   */
 
 /* μ-constants:  μ[i] = ⌊2³² / (PARAM_N−i)⌋  */
 static const uint32_t m_val[TABLE_SZ] = {
   74517,74518,74520,74521,74522,74524,74525,74526,74527,74529,74530,74531,74533,
   74534,74535,74536,74538,74539,74540,74542,74543,74544,74545,74547,74548,74549,
   74551,74552,74553,74555,74556,74557,74558,74560,74561,74562,74564,74565,74566,
   74567,74569,74570,74571,74573,74574,74575,74577,74578,74579,74580,74582,74583,
   74584,74586,74587,74588,74590,74591,74592,74593,74595,74596,74597,74599,74600,
   74601,74602,74604,74605,74606,74608,74609,74610,74612,74613,74614,74615,74617,
   74618,74619,74621,74622,74623,74625,74626,74627,74628,74630,74631,74632,74634,
   74635,74636,74637,74639,74640,74641,74643,74644,74645,74647,74648,74649,74650,
   74652,74653,74654,74656,74657,74658,74660,74661,74662,74663,74665,74666,74667,
   74669,74670,74671,74673,74674,74675,74676,74678,74679,74680,74682,74683,74684,
   74685,74687,74688,74689,74691,74692,74693,74695,74696,74697,74698,74700,74701,
   74702,74704,74705,74706,74708,74709
 };
 
 /* ------------ helpers -------------------------------------------------- */
 static inline uint32_t cond_sub(uint32_t r, uint32_t n)
 {
     uint32_t mask = -(r >= n);      /* 0xFFFFFFFF if r ≥ n */
     return r - (n & mask);          /* r or r−n */
 }
 
 /* Barrett reduction a mod (PARAM_N−i) */
 static inline uint32_t reduce_barrett(uint32_t a, size_t i)
 {
     uint32_t q = ((uint64_t)a * m_val[i]) >> 32;
     uint32_t n = PARAM_N - (uint32_t)i;
     uint32_t r = a - q * n;
     return cond_sub(r, n);
 }
 
 /* ------------------------ validation ----------------------------------- */
 static void validate_edge_cases(void)
 {
     for (size_t i = 0; i < TABLE_SZ; ++i) {
         uint32_t n = PARAM_N - (uint32_t)i;
 
         assert(reduce_barrett(0,       i) == 0);
         assert(reduce_barrett(n - 1,   i) == n - 1);
         assert(reduce_barrett(n,       i) == 0);
         assert(reduce_barrett(2 * n,   i) == 0);
     }
     puts("edge-case tests OK");
 }
 
 /* ------------------------ benchmark ------------------------------------ */
 int main(void)
 {
     const size_t ITERS = 10000000;
     const size_t RUNS  = 10;
 
     uint32_t *inputs = aligned_alloc(64, ITERS * sizeof(uint32_t));
     for (size_t i = 0; i < ITERS; ++i)
         inputs[i] = 0x9E3779B9u * i + 0x7F4A7C15u;
 
     validate_edge_cases();
 
     uint64_t cycles_total = 0, checksum = 0;
     for (size_t run = 0; run < RUNS; ++run) {
         unsigned junk;
         uint64_t t0 = __rdtscp(&junk);
         for (size_t j = 0; j < ITERS; ++j)
             checksum += reduce_barrett(inputs[j], j % TABLE_SZ);
         uint64_t t1 = __rdtscp(&junk);
         cycles_total += t1 - t0;
     }
 
     printf("checksum          %llu\n", (unsigned long long)checksum);
     printf("avg cycles/call   %.3f\n",
            (double)cycles_total / (RUNS * ITERS));
 
     free(inputs);
     return 0;
 }
 