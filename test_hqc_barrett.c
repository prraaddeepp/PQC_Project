/*
 * bench_hqc_barrett.c  —  full-scheme HQC-128 timing (Barrett reducer)
 *
 * Compile against liboqs that was built with -DOQS_ENABLE_KEM_HQC=ON.
 * The reducer inside vector.c is Barrett by default, so this measures
 * the reference path without any code changes.
 *
 * Usage:  ./bench_hqc_barrett   [ITERATIONS, default 1000]
 */

 #include <oqs/kem.h>

 #include <stdint.h>
 #include <stdio.h>
 #include <stdlib.h>
 #include <string.h>
 #include <time.h>
 #include <x86intrin.h>          /* for rdtscp on x86/AMD64          */
 
 #if !defined(CLOCK_MONOTONIC_RAW)
   #define CLOCK_MONOTONIC_RAW CLOCK_MONOTONIC
 #endif
 
 /* --------- cycle counter helpers ------------------------------------ */
 static inline uint64_t rdtscp(void) {
     unsigned aux;
     return __rdtscp(&aux);
 }
 
 static inline uint64_t cycles_diff(uint64_t start, uint64_t end) {
     return end - start;
 }
 
 /* --------- wall-clock helpers (nanoseconds) ------------------------- */
 static inline uint64_t nstime(void) {
     struct timespec ts;
     clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
     return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
 }
 
 /* --------- pretty-printer ------------------------------------------- */
 static void print_stats(const char *label,
                         uint64_t cyc_total, uint64_t ns_total,
                         size_t iterations) {
     printf("%-10s : %8.2f cycles  |  %8.2f ns\n",
            label,
            (double)cyc_total / iterations,
            (double)ns_total  / iterations);
 }
 
 /* ==================================================================== */
 int main(int argc, char *argv[]) {
 
     /* ---------------------------------------------------------------- */
     /* 1.  Select HQC-128 (Barrett reducer built into vector.c)         */
     /* ---------------------------------------------------------------- */
     OQS_KEM *kem = OQS_KEM_new(OQS_KEM_alg_hqc_128);
     if (!kem) {
         fprintf(stderr, "Could not init OQS_KEM_alg_hqc_128\n");
         return EXIT_FAILURE;
     }
 
     /* optional CLI arg: iteration count */
     const size_t R = (argc >= 2) ? strtoul(argv[1], NULL, 10) : 10000;
 
     /* ---------------------------------------------------------------- */
     /* 2.  Allocate temporary buffers                                   */
     /* ---------------------------------------------------------------- */
     uint8_t *pk  = malloc(kem->length_public_key);
     uint8_t *sk  = malloc(kem->length_secret_key);
     uint8_t *ct  = malloc(kem->length_ciphertext);
     uint8_t *ss1 = malloc(kem->length_shared_secret);
     uint8_t *ss2 = malloc(kem->length_shared_secret);
     if (!pk || !sk || !ct || !ss1 || !ss2) {
         fprintf(stderr, "malloc failed\n");
         return EXIT_FAILURE;
     }
 
     /* ---------------------------------------------------------------- */
     /* 3.  Benchmark loop                                               */
     /* ---------------------------------------------------------------- */
     uint64_t cyc_key = 0, cyc_enc = 0, cyc_dec = 0;
     uint64_t ns_key  = 0, ns_enc = 0, ns_dec = 0;
 
     for (size_t i = 0; i < R; ++i) {
         /* ---- key generation ---------------------------------------- */
         uint64_t c0 = rdtscp();
         uint64_t t0 = nstime();
         if (OQS_KEM_keypair(kem, pk, sk) != OQS_SUCCESS) {
             fprintf(stderr, "keypair failed at iter %zu\n", i);
             return EXIT_FAILURE;
         }
         uint64_t t1 = nstime();
         uint64_t c1 = rdtscp();
         cyc_key += cycles_diff(c0, c1);
         ns_key  += t1 - t0;
 
         /* ---- encapsulation ----------------------------------------- */
         c0 = rdtscp();  t0 = nstime();
         if (OQS_KEM_encaps(kem, ct, ss1, pk) != OQS_SUCCESS) {
             fprintf(stderr, "encaps failed at iter %zu\n", i);
             return EXIT_FAILURE;
         }
         t1 = nstime();  c1 = rdtscp();
         cyc_enc += cycles_diff(c0, c1);
         ns_enc  += t1 - t0;
 
         /* ---- decapsulation ----------------------------------------- */
         c0 = rdtscp();  t0 = nstime();
         if (OQS_KEM_decaps(kem, ss2, ct, sk) != OQS_SUCCESS) {
             fprintf(stderr, "decaps failed at iter %zu\n", i);
             return EXIT_FAILURE;
         }
         t1 = nstime();  c1 = rdtscp();
         cyc_dec += cycles_diff(c0, c1);
         ns_dec  += t1 - t0;
 
         if (memcmp(ss1, ss2, kem->length_shared_secret) != 0) {
             fprintf(stderr, "shared-secret mismatch at iter %zu\n", i);
             return EXIT_FAILURE;
         }
     }
 
     /* ---------------------------------------------------------------- */
     /* 4.  Results                                                      */
     /* ---------------------------------------------------------------- */
     puts("HQC-128  (reference Barrett reduction)\n"
          "---------------------------------------");
     printf("Iterations   : %zu\n\n", R);
     print_stats("keygen",  cyc_key, ns_key, R);
     print_stats("encap",   cyc_enc, ns_enc, R);
     print_stats("decap",   cyc_dec, ns_dec, R);
 
     /* ---------------------------------------------------------------- */
     /* 5.  Cleanup                                                      */
     /* ---------------------------------------------------------------- */
     free(pk); free(sk); free(ct); free(ss1); free(ss2);
     OQS_KEM_free(kem);
     return EXIT_SUCCESS;
 }
 