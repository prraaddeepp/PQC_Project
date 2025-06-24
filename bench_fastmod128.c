/*****************************************************************
 * FastMod / libdivide reducer benchmark  (floor-reciprocal, 32-bit)
 *   – identical harness to your Shoup & Barrett tests
 *****************************************************************/
 #include <stdio.h>
 #include <stdint.h>
 #include <x86intrin.h>
 #include <assert.h>
 #include <stdlib.h>
 
 #define PARAM_N 17669u        /* HQC-128 */
 #define TABLE_SZ 75           /* ω_max */
 
 /* = floor(2^64 / (PARAM_N - i))  = R_val[i] - 1  because 2^64 is
    not divisible by any n in [17595..17669]                                          */
 static const uint64_t M_val[TABLE_SZ] = {
   1044017435831657ULL,1044076526698525ULL,1044135624254799ULL,1044194728501616ULL,
   1044253839440110ULL,1044312957071419ULL,1044372081396679ULL,1044431212417028ULL,
   1044490350133602ULL,1044549494547539ULL,1044608645659977ULL,1044667803472055ULL,
   1044726967984909ULL,1044786139199680ULL,1044845317117504ULL,1044904501739523ULL,
   1044963693066875ULL,1045022891100699ULL,1045082095842136ULL,1045141307292325ULL,
   1045200525452408ULL,1045259750323523ULL,1045318981906814ULL,1045378220203420ULL,
   1045437465214482ULL,1045496716941144ULL,1045555975384546ULL,1045615240545831ULL,
   1045674512426140ULL,1045733791026618ULL,1045793076348406ULL,1045852368392649ULL,
   1045911667160489ULL,1045970972653070ULL,1046030284871536ULL,1046089603817032ULL,
   1046148929490702ULL,1046208261893690ULL,1046267601027142ULL,1046326946892203ULL,
   1046386299490019ULL,1046445658821735ULL,1046505024888497ULL,1046564397691452ULL,
   1046623777231747ULL,1046683163510528ULL,1046742556528942ULL,1046801956288137ULL,
   1046861362789260ULL,1046920776033459ULL,1046980196021882ULL,1047039622755678ULL,
   1047099056235996ULL,1047158496463984ULL,1047217943440791ULL,1047277397167568ULL,
   1047336857645463ULL,1047396324875627ULL,1047455798859210ULL,1047515279597362ULL,
   1047574767091234ULL,1047634261341978ULL,1047693762350744ULL,1047753270118684ULL,
   1047812784646949ULL,1047872305936693ULL,1047931833989067ULL,1047991368805223ULL,
   1048050910386316ULL,1048110458733497ULL,1048170013847920ULL,1048229575730739ULL,
   1048289144383107ULL,1048348719806180ULL,1048408302001111ULL
 };
 
 /* branch-free FastMod reduction */
 static inline uint32_t reduce_fastmod(uint32_t a, size_t i)
 {
     uint64_t q = (uint64_t)((__uint128_t)a * M_val[i] >> 64);
     uint32_t n = PARAM_N - (uint32_t)i;
     uint32_t r = a - q * n;
     uint32_t mask = -(r >= n);     /* 0xFFFFFFFF if r ≥ n else 0 */
     return r - (n & mask);         /* at most one subtract       */
 }
 
 /* ---------- self-test on edge cases ---------- */
 static void validate_edge_cases(void)
 {
     for (size_t i = 0; i < TABLE_SZ; ++i) {
         uint32_t n = PARAM_N - (uint32_t)i;
         assert(reduce_fastmod(0, i)          == 0);
         assert(reduce_fastmod(n-1, i)        == n-1);
         assert(reduce_fastmod(n, i)          == 0);
         assert(reduce_fastmod(2*n, i)        == 0);
     }
     puts("edge-case tests OK");
 }
 
 int main(void)
 {
     const size_t ITERS = 10000000, RUNS = 10;
     uint64_t tot_cycles = 0, sum = 0;
 
     /* deterministic “random” inputs */
     uint32_t *inp = aligned_alloc(64, ITERS*sizeof(uint32_t));
     for (size_t i = 0; i < ITERS; ++i)
         inp[i] = 0x9E3779B9u * i + 0x7F4A7C15u;
 
     validate_edge_cases();
 
     for (size_t run = 0; run < RUNS; ++run) {
         unsigned junk;
         uint64_t st = __rdtscp(&junk);
         for (size_t i = 0; i < ITERS; ++i)
             sum += reduce_fastmod(inp[i], i % TABLE_SZ);
         uint64_t en = __rdtscp(&junk);
         tot_cycles += en - st;
     }
 
     printf("checksum          %llu\n", (unsigned long long)sum);
     printf("avg cycles/call   %.3f\n",
            (double)tot_cycles / (RUNS * ITERS));
 
     free(inp);
     return 0;
 }
 