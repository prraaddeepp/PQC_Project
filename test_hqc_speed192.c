/*
 * test_hqc_barrett.c  —  HQC-256 timing (Barrett reducer),
 * batch-mean ± population SD over calls in C.
 */

 #include <oqs/kem.h>

 #include <stdint.h>
 #include <stdio.h>
 #include <stdlib.h>
 #include <string.h>
 #include <time.h>
 #include <math.h>      // for sqrt()
 #include <x86intrin.h> // for rdtscp()
 
 /* --------------------------------------------------------------------
  *  platform-specific includes for pinning thread to one core
  * ------------------------------------------------------------------*/
 #if defined(__linux__)
   #include <pthread.h>
   #include <sched.h>
 #elif defined(__APPLE__)
   #include <pthread.h>
   #include <mach/mach.h>
 #endif
 
 /* pin_this_thread(core_or_tag):
  *  - On Linux: hard-affinity to logical CPU #core_or_tag.
  *  - On macOS: best-effort tag hint (threads with same tag co-locate).
  *  - Elsewhere: no-op.
  */
 static void pin_this_thread(int core_or_tag) {
 #if defined(__linux__)
     cpu_set_t set;
     CPU_ZERO(&set);
     CPU_SET(core_or_tag, &set);
     if (sched_setaffinity(0, sizeof(set), &set) != 0) {
         perror("sched_setaffinity");
     }
 #elif defined(__APPLE__)
     thread_affinity_policy_data_t policy = { .affinity_tag = core_or_tag ? core_or_tag : 1 };
     kern_return_t kr = thread_policy_set(
         pthread_mach_thread_np(pthread_self()),
         THREAD_AFFINITY_POLICY,
         (thread_policy_t)&policy,
         THREAD_AFFINITY_POLICY_COUNT);
     if (kr != KERN_SUCCESS) {
         fprintf(stderr, "thread_policy_set failed: %d\n", kr);
     }
 #else
     (void)core_or_tag;
 #endif
 }
 /* ------------------------------------------------------------------ */
 
 #if !defined(CLOCK_MONOTONIC_RAW)
   #define CLOCK_MONOTONIC_RAW CLOCK_MONOTONIC
 #endif
 
 static inline uint64_t rdtscp(void) {
     unsigned aux;
     return __rdtscp(&aux);
 }
 
 static inline uint64_t nstime(void) {
     struct timespec ts;
     clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
     return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
 }
 
 /* Compute mean and population SD over `n` samples in `data[]`. */
 static void compute_mean_sd(const double *data, size_t n,
                             double *out_mean, double *out_sd) {
     double sum = 0.0, sumsq = 0.0;
     for (size_t i = 0; i < n; i++) {
         sum   += data[i];
         sumsq += data[i] * data[i];
     }
     double mean = sum / (double)n;
     double var  = sumsq / (double)n - mean * mean;
     *out_mean = mean;
     *out_sd   = (var > 0.0 ? sqrt(var) : 0.0);
 }
 
 /* Print mean ± SD in cycles and nanoseconds */
 static void print_stats_sd(const char *label,
                            double mean_cyc, double sd_cyc,
                            double mean_ns,  double sd_ns) {
     printf("%-10s : %8.2f ± %8.2f cycles  |  %8.2f ± %8.2f ns\n",
            label,
            mean_cyc, sd_cyc,
            mean_ns,  sd_ns);
 }
 
 int main(int argc, char *argv[]) {
     /* Pin this thread to core/tag 7 */
     pin_this_thread(7);
 
     /* batch settings: calls per batch, and number of batches */
     const size_t BATCH   = (argc >= 2 ? strtoul(argv[1], NULL, 10) : 1000);
     const size_t REPEATS = (argc >= 3 ? strtoul(argv[2], NULL, 10) : 10);
 
     printf("HQC-256 batch benchmark (%zu×%zu = %zu calls)\n",
            REPEATS, BATCH, BATCH * REPEATS);
     printf("---------------------------------------------\n");
 
     OQS_KEM *kem = OQS_KEM_new(OQS_KEM_alg_hqc_256);
     if (!kem) {
         fprintf(stderr, "init HQC-256 failed\n");
         return EXIT_FAILURE;
     }
 
     uint8_t *pk  = malloc(kem->length_public_key);
     uint8_t *sk  = malloc(kem->length_secret_key);
     uint8_t *ct  = malloc(kem->length_ciphertext);
     uint8_t *ss1 = malloc(kem->length_shared_secret);
     uint8_t *ss2 = malloc(kem->length_shared_secret);
     if (!pk || !sk || !ct || !ss1 || !ss2) {
         fprintf(stderr, "malloc failed\n");
         return EXIT_FAILURE;
     }
 
     double *key_cyc = calloc(REPEATS, sizeof(double));
     double *enc_cyc = calloc(REPEATS, sizeof(double));
     double *dec_cyc = calloc(REPEATS, sizeof(double));
     double *key_ns  = calloc(REPEATS, sizeof(double));
     double *enc_ns  = calloc(REPEATS, sizeof(double));
     double *dec_ns  = calloc(REPEATS, sizeof(double));
     if (!key_cyc || !enc_cyc || !dec_cyc || !key_ns || !enc_ns || !dec_ns) {
         fprintf(stderr, "calloc failed\n");
         return EXIT_FAILURE;
     }
 
     for (size_t r = 0; r < REPEATS; r++) {
         uint64_t sum_key_c = 0, sum_enc_c = 0, sum_dec_c = 0;
         uint64_t sum_key_n = 0, sum_enc_n = 0, sum_dec_n = 0;
 
         for (size_t i = 0; i < BATCH; i++) {
             /* keygen */
             uint64_t c0 = rdtscp(), t0 = nstime();
             OQS_KEM_keypair(kem, pk, sk);
             sum_key_c += rdtscp() - c0;
             sum_key_n += nstime()  - t0;
 
             /* encaps */
             c0 = rdtscp(); t0 = nstime();
             OQS_KEM_encaps(kem, ct, ss1, pk);
             sum_enc_c += rdtscp() - c0;
             sum_enc_n += nstime()  - t0;
 
             /* decaps */
             c0 = rdtscp(); t0 = nstime();
             OQS_KEM_decaps(kem, ss2, ct, sk);
             sum_dec_c += rdtscp() - c0;
             sum_dec_n += nstime()  - t0;
 
             /* correctness */
             if (memcmp(ss1, ss2, kem->length_shared_secret) != 0) {
                 fprintf(stderr, "mismatch at batch %zu iter %zu\n", r, i);
                 return EXIT_FAILURE;
             }
         }
 
         key_cyc[r] = (double)sum_key_c / BATCH;
         enc_cyc[r] = (double)sum_enc_c / BATCH;
         dec_cyc[r] = (double)sum_dec_c / BATCH;
         key_ns [r] = (double)sum_key_n / BATCH;
         enc_ns [r] = (double)sum_enc_n / BATCH;
         dec_ns [r] = (double)sum_dec_n / BATCH;
     }
 
     /* compute mean and SD over the REPEATS batch-means */
     double mean_key_c, sd_key_c, mean_enc_c, sd_enc_c, mean_dec_c, sd_dec_c;
     double mean_key_n, sd_key_n, mean_enc_n, sd_enc_n, mean_dec_n, sd_dec_n;
 
     compute_mean_sd(key_cyc, REPEATS,  &mean_key_c, &sd_key_c);
     compute_mean_sd(enc_cyc, REPEATS,  &mean_enc_c, &sd_enc_c);
     compute_mean_sd(dec_cyc, REPEATS,  &mean_dec_c, &sd_dec_c);
     compute_mean_sd(key_ns,  REPEATS,  &mean_key_n, &sd_key_n);
     compute_mean_sd(enc_ns,  REPEATS,  &mean_enc_n, &sd_enc_n);
     compute_mean_sd(dec_ns,  REPEATS,  &mean_dec_n, &sd_dec_n);
 
     puts("Results (mean ± sd over batches)\n"
          "--------------------------------");
     print_stats_sd("keygen", mean_key_c, sd_key_c, mean_key_n, sd_key_n);
     print_stats_sd("encap",  mean_enc_c, sd_enc_c, mean_enc_n, sd_enc_n);
     print_stats_sd("decap",  mean_dec_c, sd_dec_c, mean_dec_n, sd_dec_n);
 
     /* cleanup */
     free(pk); free(sk); free(ct); free(ss1); free(ss2);
     free(key_cyc); free(enc_cyc); free(dec_cyc);
     free(key_ns);  free(enc_ns);  free(dec_ns);
     OQS_KEM_free(kem);
 
     return EXIT_SUCCESS;
 }
 