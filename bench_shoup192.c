/*****************************************************************
 * Accurate Shoup Reciprocal Reducer Benchmark  –  HQC-192
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
 
 #define PARAM_N   35851u      /* n for HQC-192                 */
 #define TABLE_SZ  114         /* ω_max                         */
 
 /* R_val[i] = ceil(2^64 / (PARAM_N-i)) */
 static const uint64_t R_val[TABLE_SZ] = {
   514539178090139ULL, 514553530647408ULL, 514567884005400ULL, 514582238164181ULL,
   514596593123820ULL, 514610948884382ULL, 514625305445936ULL, 514639662808547ULL,
   514654020972284ULL, 514668379937212ULL, 514682739703400ULL, 514697100270914ULL,
   514711461639822ULL, 514725823810189ULL, 514740186782085ULL, 514754550555575ULL,
   514768915130726ULL, 514783280507606ULL, 514797646686283ULL, 514812013666822ULL,
   514826381449292ULL, 514840750033759ULL, 514855119420290ULL, 514869489608953ULL,
   514883860599815ULL, 514898232392943ULL, 514912604988404ULL, 514926978386265ULL,
   514941352586594ULL, 514955727589458ULL, 514970103394924ULL, 514984480003059ULL,
   514998857413930ULL, 515013235627605ULL, 515027614644151ULL, 515041994463635ULL,
   515056375086125ULL, 515070756511687ULL, 515085138740389ULL, 515099521772299ULL,
   515113905607483ULL, 515128290246009ULL, 515142675687943ULL, 515157061933355ULL,
   515171448982310ULL, 515185836834876ULL, 515200225491120ULL, 515214614951111ULL,
   515229005214914ULL, 515243396282598ULL, 515257788154229ULL, 515272180829876ULL,
   515286574309606ULL, 515300968593485ULL, 515315363681581ULL, 515329759573963ULL,
   515344156270696ULL, 515358553771849ULL, 515372952077489ULL, 515387351187684ULL,
   515401751102500ULL, 515416151822005ULL, 515430553346268ULL, 515444955675354ULL,
   515459358809332ULL, 515473762748269ULL, 515488167492233ULL, 515502573041291ULL,
   515516979395511ULL, 515531386554960ULL, 515545794519705ULL, 515560203289815ULL,
   515574612865356ULL, 515589023246396ULL, 515603434433004ULL, 515617846425245ULL,
   515632259223188ULL, 515646672826901ULL, 515661087236451ULL, 515675502451906ULL,
   515689918473332ULL, 515704335300799ULL, 515718752934372ULL, 515733171374121ULL,
   515747590620113ULL, 515762010672414ULL, 515776431531094ULL, 515790853196219ULL,
   515805275667857ULL, 515819698946076ULL, 515834123030943ULL, 515848547922527ULL,
   515862973620895ULL, 515877400126114ULL, 515891827438252ULL, 515906255557377ULL,
   515920684483557ULL, 515935114216859ULL, 515949544757351ULL, 515963976105101ULL,
   515978408260176ULL, 515992841222645ULL, 516007274992575ULL, 516021709570034ULL,
   516036144955089ULL, 516050581147809ULL, 516065018148260ULL, 516079455956512ULL,
   516093894572632ULL, 516108333996686ULL, 516122774228745ULL, 516137215268874ULL,
   516151657117143ULL, 516166099773618ULL
 };
 
 /* ---------------- reduction ---------------- */
 static inline uint32_t reduce_shoup(uint32_t a, size_t i)
 {
     unsigned __int128 prod = (unsigned __int128)a * R_val[i];
     uint64_t q = (uint64_t)(prod >> 64);
     uint32_t n = PARAM_N - (uint32_t)i;
     uint32_t r = a - (uint32_t)q * n;
     uint32_t mask = -(r >= n);
     return r - (n & mask);          /* constant-time final subtract */
 }
 
 /* ---------------- self-test ---------------- */
 static void validate_edge_cases(void)
 {
     for (size_t i = 0; i < TABLE_SZ; ++i) {
         uint32_t n = PARAM_N - (uint32_t)i;
         assert(reduce_shoup(0,      i) == 0);
         assert(reduce_shoup(n - 1,  i) == n - 1);
         assert(reduce_shoup(n,      i) == 0);
         assert(reduce_shoup(2U*n,   i) == 0);
     }
     puts("edge-case tests OK");
 }
 
 int main(void)
 {
     const size_t ITERS = 10000000, RUNS = 10;
     uint64_t total_cycles = 0, sum = 0;
 
     uint32_t *inputs = aligned_alloc(64, ITERS * sizeof(uint32_t));
     for (size_t i = 0; i < ITERS; ++i)
         inputs[i] = 0x9e3779b9u * (uint32_t)i + 0x7f4a7c15u;
 
     validate_edge_cases();
 
     for (size_t run = 0; run < RUNS; ++run) {
         unsigned junk;
         uint64_t start = __rdtscp(&junk);
         for (size_t i = 0; i < ITERS; ++i)
             sum += reduce_shoup(inputs[i], i % TABLE_SZ);
         uint64_t end = __rdtscp(&junk);
         total_cycles += end - start;
     }
 
     printf("Checksum          = %llu\n", (unsigned long long)sum);
     printf("Avg cycles/call   = %.3f\n",
            (double)total_cycles / (RUNS * ITERS));
 
     free(inputs);
     return 0;
 }
 