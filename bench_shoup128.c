/*****************************************************************
 * Accurate Shoup Reciprocal Reducer Benchmark
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
 
 // Precomputed R_val[i] = ceil(2^64 / (PARAM_N - i))
 static const uint64_t R_val[75] = {
    1044017435831658, 1044076526698526, 1044135624254800, 1044194728501617, 1044253839440111, 1044312957071420, 1044372081396680, 1044431212417029, 1044490350133603, 1044549494547540, 1044608645659978, 1044667803472056, 1044726967984910, 1044786139199681, 1044845317117505, 1044904501739524, 1044963693066876, 1045022891100700, 1045082095842137, 1045141307292326, 1045200525452409, 1045259750323524, 1045318981906815, 1045378220203421, 1045437465214483, 1045496716941145, 1045555975384547, 1045615240545832, 1045674512426141, 1045733791026619, 1045793076348407, 1045852368392650, 1045911667160490, 1045970972653071, 1046030284871537, 1046089603817033, 1046148929490703, 1046208261893691, 1046267601027143, 1046326946892204, 1046386299490020, 1046445658821736, 1046505024888498, 1046564397691453, 1046623777231748, 1046683163510529, 1046742556528943, 1046801956288138, 1046861362789261, 1046920776033460, 1046980196021883, 1047039622755679, 1047099056235997, 1047158496463985, 1047217943440792, 1047277397167569, 1047336857645464, 1047396324875628, 1047455798859211, 1047515279597363, 1047574767091235, 1047634261341979, 1047693762350745, 1047753270118685, 1047812784646950, 1047872305936694, 1047931833989068, 1047991368805224, 1048050910386317, 1048110458733498, 1048170013847921, 1048229575730740, 1048289144383108, 1048348719806181, 1048408302001112
 };
 
 static inline uint32_t reduce_shoup(uint32_t a, size_t i) {
     // Compute q = floor(a * (2^64/n)) via high 64 bits of 128-bit product
     unsigned __int128 prod = (unsigned __int128)a * R_val[i];
     uint64_t q = (uint64_t)(prod >> 64);
     uint32_t n = PARAM_N - (uint32_t)i;
     uint32_t r = a - (uint32_t)q * n;
     // final conditional subtract, constant-time
     uint32_t mask = -(r >= n);
     return r - (n & mask);
 }
 
 static void validate_edge_cases() {
     // cover both largest and smallest moduli
     assert(reduce_shoup(0, 0) == 0);
     assert(reduce_shoup(PARAM_N-1, 0) == PARAM_N-1);
     assert(reduce_shoup(PARAM_N, 0) == 0);
     assert(reduce_shoup(2*PARAM_N, 0) == 0);
 
     // smallest divisor = PARAM_N-74
     assert(reduce_shoup(0, 74) == 0);
     assert(reduce_shoup(PARAM_N-75, 74) == PARAM_N-75);
     assert(reduce_shoup(PARAM_N-74, 74) == 0);
 }
 
 int main(void) {
     const size_t ITERS = 10000000, RUNS = 10;
     uint64_t sum = 0, total_cycles = 0;
 
     // Pre-generate inputs
     uint32_t *inputs = aligned_alloc(64, ITERS * sizeof(uint32_t));
     for (size_t i = 0; i < ITERS; i++)
         inputs[i] = 0x9e3779b9u * (uint32_t)i + 0x7f4a7c15u;
 
     validate_edge_cases();
 
     // Benchmark
     for (size_t run = 0; run < RUNS; run++) {
         unsigned junk;
         uint64_t start = __rdtscp(&junk);
         for (size_t i = 0; i < ITERS; i++)
             sum += reduce_shoup(inputs[i], i % 75);
         uint64_t end = __rdtscp(&junk);
         total_cycles += (end - start);
     }
 
     printf("Checksum = %llu\n", (unsigned long long)sum);
     printf("Avg cycles/call = %.3f\n",
            (double)total_cycles/(RUNS*ITERS));
 
     free(inputs);
     return 0;
 }
 