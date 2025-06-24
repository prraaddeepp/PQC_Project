/*****************************************************************
 * FastMod / libdivide reducer benchmark  (floor-reciprocal, 32-bit)
 *   – identical harness to your Shoup & Barrett tests
 *****************************************************************/
 #include <stdio.h>
 #include <stdint.h>
 #include <x86intrin.h>
 #include <assert.h>
 #include <stdlib.h>
 
 #define PARAM_N 35851u        /* HQC-128 */
 #define TABLE_SZ 114           /* ω_max */
                                         
    static const uint64_t M_val[TABLE_SZ] = {
        514539178090138ULL,514553530647407ULL,514567884005399ULL,514582238164180ULL,
        514596593123819ULL,514610948884381ULL,514625305445935ULL,514639662808546ULL,
        514654020972283ULL,514668379937211ULL,514682739703399ULL,514697100270913ULL,
        514711461639821ULL,514725823810188ULL,514740186782084ULL,514754550555574ULL,
        514768915130725ULL,514783280507605ULL,514797646686282ULL,514812013666821ULL,
        514826381449291ULL,514840750033758ULL,514855119420289ULL,514869489608952ULL,
        514883860599814ULL,514898232392942ULL,514912604988403ULL,514926978386264ULL,
        514941352586593ULL,514955727589457ULL,514970103394923ULL,514984480003058ULL,
        514998857413929ULL,515013235627604ULL,515027614644150ULL,515041994463634ULL,
        515056375086124ULL,515070756511686ULL,515085138740388ULL,515099521772298ULL,
        515113905607482ULL,515128290246008ULL,515142675687942ULL,515157061933354ULL,
        515171448982309ULL,515185836834875ULL,515200225491119ULL,515214614951110ULL,
        515229005214913ULL,515243396282597ULL,515257788154228ULL,515272180829875ULL,
        515286574309605ULL,515300968593484ULL,515315363681580ULL,515329759573962ULL,
        515344156270695ULL,515358553771848ULL,515372952077488ULL,515387351187683ULL,
        515401751102499ULL,515416151822004ULL,515430553346267ULL,515444955675353ULL,
        515459358809331ULL,515473762748268ULL,515488167492232ULL,515502573041290ULL,
        515516979395510ULL,515531386554959ULL,515545794519704ULL,515560203289814ULL,
        515574612865355ULL,515589023246395ULL,515603434433003ULL,515617846425244ULL,
        515632259223187ULL,515646672826900ULL,515661087236450ULL,515675502451905ULL,
        515689918473331ULL,515704335300798ULL,515718752934371ULL,515733171374120ULL,
        515747590620112ULL,515762010672413ULL,515776431531093ULL,515790853196218ULL,
        515805275667856ULL,515819698946075ULL,515834123030942ULL,515848547922526ULL,
        515862973620894ULL,515877400126113ULL,515891827438251ULL,515906255557376ULL,
        515920684483556ULL,515935114216858ULL,515949544757350ULL,515963976105100ULL,
        515978408260175ULL,515992841222644ULL,516007274992574ULL,516021709570033ULL,
        516036144955088ULL,516050581147808ULL,516065018148259ULL,516079455956511ULL,
        516093894572631ULL,516108333996685ULL,516122774228744ULL,516137215268873ULL,
        516151657117142ULL,516166099773617ULL
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
 