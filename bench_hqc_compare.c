// tests/bench_hqc_compare.c
// Paired comparison: Barrett vs. Mul-Shift HQC-192 reducers

#include <oqs/kem.h>
#include "hqc_reducer.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <x86intrin.h>

#if defined(__APPLE__)
  #include <pthread.h>
  #include <mach/mach.h>
#endif

//-----------------------------------------------------------------------------
// pin_this_thread(core): hard-affinity (Linux) or affinity hint (macOS)
static void pin_this_thread(int core) {
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core, &set);
    sched_setaffinity(0, sizeof(set), &set);
#elif defined(__APPLE__)
    thread_affinity_policy_data_t pol = { .affinity_tag = core ? core : 1 };
    thread_policy_set(pthread_mach_thread_np(pthread_self()),
                      THREAD_AFFINITY_POLICY,
                      (thread_policy_t)&pol,
                      THREAD_AFFINITY_POLICY_COUNT);
#else
    (void)core;
#endif
}

//-----------------------------------------------------------------------------
// High-resolution timer
static inline uint64_t rdtscp(void) {
    unsigned aux;
    return __rdtscp(&aux);
}

// Compute mean and population SD over n samples in data[]
static void compute_mean_sd(const double *data, size_t n,
                            double *out_mean, double *out_sd) {
    double sum = 0, sumsq = 0;
    for (size_t i = 0; i < n; i++) {
        sum   += data[i];
        sumsq += data[i] * data[i];
    }
    double mean = sum / (double)n;
    double var  = sumsq / (double)n - mean * mean;
    *out_mean = mean;
    *out_sd   = (var > 0 ? sqrt(var) : 0);
}

//-----------------------------------------------------------------------------
// Run one batch of BATCH calls, accumulate cycles in sums
static void run_batch(OQS_KEM *kem,
                      uint8_t *pk, uint8_t *sk,
                      uint8_t *ct, uint8_t *ss1, uint8_t *ss2,
                      size_t BATCH,
                      uint64_t *sum_key,
                      uint64_t *sum_enc,
                      uint64_t *sum_dec) {
    for (size_t i = 0; i < BATCH; i++) {
        uint64_t t0 = rdtscp();
        OQS_KEM_keypair(kem, pk, sk);
        *sum_key += rdtscp() - t0;

        t0 = rdtscp();
        OQS_KEM_encaps(kem, ct, ss1, pk);
        *sum_enc += rdtscp() - t0;

        t0 = rdtscp();
        OQS_KEM_decaps(kem, ss2, ct, sk);
        *sum_dec += rdtscp() - t0;

        if (memcmp(ss1, ss2, kem->length_shared_secret) != 0) {
            fprintf(stderr, "shared-secret mismatch\n");
            exit(1);
        }
    }
}

//-----------------------------------------------------------------------------
// Main: ./bench_hqc_compare [BATCH] [REPEATS]
int main(int argc, char **argv) {
    size_t BATCH   = (argc > 1 ? strtoul(argv[1], NULL, 10) : 1000);
    size_t REPEATS = (argc > 2 ? strtoul(argv[2], NULL, 10) : 10);

    pin_this_thread(7);

    printf("Paired HQC-192 bench: %zu batches of %zu calls (Barrett ↔ MulShift)\n",
           REPEATS, BATCH);

    OQS_KEM *kem = OQS_KEM_new(OQS_KEM_alg_hqc_192);
    if (!kem) {
        fprintf(stderr, "OQS_KEM_new failed\n");
        return 1;
    }

    // Allocate buffers
    uint8_t *pk  = malloc(kem->length_public_key);
    uint8_t *sk  = malloc(kem->length_secret_key);
    uint8_t *ct  = malloc(kem->length_ciphertext);
    uint8_t *ss1 = malloc(kem->length_shared_secret);
    uint8_t *ss2 = malloc(kem->length_shared_secret);

    // Arrays for raw per-call cycles
    double *bar_key = calloc(REPEATS, sizeof(double));
    double *bar_enc = calloc(REPEATS, sizeof(double));
    double *bar_dec = calloc(REPEATS, sizeof(double));
    double *mul_key = calloc(REPEATS, sizeof(double));
    double *mul_enc = calloc(REPEATS, sizeof(double));
    double *mul_dec = calloc(REPEATS, sizeof(double));

    // Array for differences
    double *diff_key = calloc(REPEATS, sizeof(double));
    double *diff_enc = calloc(REPEATS, sizeof(double));
    double *diff_dec = calloc(REPEATS, sizeof(double));

    for (size_t r = 0; r < REPEATS; r++) {
        uint64_t kA = 0, eA = 0, dA = 0;
        uint64_t kB = 0, eB = 0, dB = 0;
        if (r & 1) {
            // then Barrett
            hqc192_use_barrett();
            run_batch(kem, pk, sk, ct, ss1, ss2, BATCH, &kA, &eA, &dA);
            bar_key[r] = (double)kA / (double)BATCH;
            bar_enc[r] = (double)eA / (double)BATCH;
            bar_dec[r] = (double)dA / (double)BATCH;

            hqc192_use_mulshift();
            run_batch(kem, pk, sk, ct, ss1, ss2, BATCH, &kB, &eB, &dB);
            mul_key[r] = (double)kB / (double)BATCH;
            mul_enc[r] = (double)eB / (double)BATCH;
            mul_dec[r] = (double)dB / (double)BATCH;
            
            // odd batch: Mul-Shift first
        } else {
            // then Mul-Shift
            hqc192_use_mulshift();
            run_batch(kem, pk, sk, ct, ss1, ss2, BATCH, &kB, &eB, &dB);
            mul_key[r] = (double)kB / (double)BATCH;
            mul_enc[r] = (double)eB / (double)BATCH;
            mul_dec[r] = (double)dB / (double)BATCH;
            
            hqc192_use_barrett();
            run_batch(kem, pk, sk, ct, ss1, ss2, BATCH, &kA, &eA, &dA);
            bar_key[r] = (double)kA / (double)BATCH;
            bar_enc[r] = (double)eA / (double)BATCH;
            bar_dec[r] = (double)dA / (double)BATCH;
            

        }

        // Differences
        diff_key[r] = bar_key[r] - mul_key[r];
        diff_enc[r] = bar_enc[r] - mul_enc[r];
        diff_dec[r] = bar_dec[r] - mul_dec[r];
    }

    // Compute stats for raw means
    double m_bar_k, sd_bar_k, m_mul_k, sd_mul_k;
    double m_bar_e, sd_bar_e, m_mul_e, sd_mul_e;
    double m_bar_d, sd_bar_d, m_mul_d, sd_mul_d;
    compute_mean_sd(bar_key, REPEATS, &m_bar_k, &sd_bar_k);
    compute_mean_sd(mul_key, REPEATS, &m_mul_k, &sd_mul_k);
    compute_mean_sd(bar_enc, REPEATS, &m_bar_e, &sd_bar_e);
    compute_mean_sd(mul_enc, REPEATS, &m_mul_e, &sd_mul_e);
    compute_mean_sd(bar_dec, REPEATS, &m_bar_d, &sd_bar_d);
    compute_mean_sd(mul_dec, REPEATS, &m_mul_d, &sd_mul_d);

    // Compute stats for differences
    double mean_k, sd_k, mean_e, sd_e, mean_d, sd_d;
    compute_mean_sd(diff_key, REPEATS, &mean_k, &sd_k);
    compute_mean_sd(diff_enc, REPEATS, &mean_e, &sd_e);
    compute_mean_sd(diff_dec, REPEATS, &mean_d, &sd_d);

    printf("\nRaw per-call cycles:\n");
    printf(" Barrett keygen : %8.2f ± %6.2f cycles\n", m_bar_k, sd_bar_k);
    printf(" MulShift keygen: %8.2f ± %6.2f cycles\n", m_mul_k, sd_mul_k);
    printf(" Barrett encap  : %8.2f ± %6.2f cycles\n", m_bar_e, sd_bar_e);
    printf(" MulShift encap : %8.2f ± %6.2f cycles\n", m_mul_e, sd_mul_e);
    printf(" Barrett decap  : %8.2f ± %6.2f cycles\n", m_bar_d, sd_bar_d);
    printf(" MulShift decap : %8.2f ± %6.2f cycles\n", m_mul_d, sd_mul_d);

    printf("\nΔ per-call cycles (Barrett − MulShift):\n");
    printf(" keygen : %8.2f ± %6.2f cycles  (%+.2f %%)\n",
           mean_k, sd_k, 100.0 * mean_k / m_mul_k);
    printf(" encap  : %8.2f ± %6.2f cycles  (%+.2f %%)\n",
           mean_e, sd_e, 100.0 * mean_e / m_mul_e);
    printf(" decap  : %8.2f ± %6.2f cycles  (%+.2f %%)\n",
           mean_d, sd_d, 100.0 * mean_d / m_mul_d);

    // cleanup
    free(pk); free(sk); free(ct); free(ss1); free(ss2);
    free(bar_key); free(bar_enc); free(bar_dec);
    free(mul_key); free(mul_enc); free(mul_dec);
    free(diff_key); free(diff_enc); free(diff_dec);
    OQS_KEM_free(kem);
    return 0;
}
