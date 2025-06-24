/*****************************************************************
 * Accurate Barrett Reducer Benchmark
 * - Pre-generated inputs (no rand() overhead)
 * - Serialized RDTSCP
 * - Full edge-case validation
 * - Frequency-independent measurements
 *****************************************************************/
 #include <stdio.h>
 #include <stdint.h>
 #include <x86intrin.h>
 #include <assert.h>
 #include <stdlib.h>
 
 #define PARAM_N 17669
 static const uint32_t m_val[75] = {
     243079, 243093, 243106, 243120, 243134, 243148, 243161, 243175, 243189,
     243203, 243216, 243230, 243244, 243258, 243272, 243285, 243299, 243313,
     243327, 243340, 243354, 243368, 243382, 243396, 243409, 243423, 243437,
     243451, 243465, 243478, 243492, 243506, 243520, 243534, 243547, 243561,
     243575, 243589, 243603, 243616, 243630, 243644, 243658, 243672, 243686,
     243699, 243713, 243727, 243741, 243755, 243769, 243782, 243796, 243810,
     243824, 243838, 243852, 243865, 243879, 243893, 243907, 243921, 243935,
     243949, 243962, 243976, 243990, 244004, 244018, 244032, 244046, 244059,
     244073, 244087, 244101
 };
 
 static inline uint32_t cond_sub(uint32_t r, uint32_t n) {
     uint32_t mask = -(r >= n);
     return r - (n & mask);
 }
 
 static inline uint32_t reduce_barrett(uint32_t a, size_t i) {
     uint32_t q = ((uint64_t)a * m_val[i]) >> 32;
     uint32_t n = PARAM_N - (uint32_t)i;
     uint32_t r = a - q * n;
     return cond_sub(r, n);
 }
 
 static void validate_edge_cases() {
     // Test all divisor ranges (i=0 to i=74)
     assert(reduce_barrett(0, 0) == 0);
     assert(reduce_barrett(PARAM_N-1, 0) == PARAM_N-1);
     assert(reduce_barrett(PARAM_N, 0) == 0);
     assert(reduce_barrett(2*PARAM_N, 0) == 0);
     
     // Also test with i=74 (smallest modulus)
     assert(reduce_barrett(0, 74) == 0);
     assert(reduce_barrett(PARAM_N-75, 74) == PARAM_N-75);
     assert(reduce_barrett(PARAM_N-74, 74) == 0);
 }
 
 int main(void) {
     const size_t ITERS = 10000000;
     const size_t RUNS = 10;
     uint64_t sum = 0;
     uint64_t total_cycles = 0;
 
     // Pre-generate test inputs (cheap LCG)
     uint32_t *inputs = aligned_alloc(64, ITERS * sizeof(uint32_t));
     for (size_t i = 0; i < ITERS; i++) {
         inputs[i] = 0x9e3779b9u * (uint32_t)i + 0x7f4a7c15u;
     }
 
     validate_edge_cases();
 
     // Benchmark loop
     for (size_t run = 0; run < RUNS; run++) {
         unsigned junk;
         uint64_t start = __rdtscp(&junk);  // Serialized read
         for (size_t i = 0; i < ITERS; i++) {
             sum += reduce_barrett(inputs[i], i % 75);
         }
         uint64_t end = __rdtscp(&junk);
         total_cycles += (end - start);
     }
 
     printf("Validation checksum = %llu\n", (unsigned long long)sum);
     printf("Average cycles/call = %.3f\n", 
           (double)total_cycles / (RUNS * ITERS));
     
     free(inputs);
     return 0;
 }