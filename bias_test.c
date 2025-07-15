// tests/bias_test_hqc128.c
// Bias test for HQC-128’s Barrett vs. MulShift reducers,
// cycling through all reduction moduli (N − i).

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <time.h>

// Pull in only the PARAM_N constant:
#include "parameters.h"   // must define: const uint32_t PARAM_N

//------------------------------------------------------------------------------
// Reduction core (from pqclean_hqc-128_clean/vector.c)

// Reciprocal constants for Barrett reduction at index i.
static const uint32_t m_val[75] = {
    243079,243093,243106,243120,243134,243148,243161,243175,243189,243203,
    243216,243230,243244,243258,243272,243285,243299,243313,243327,243340,
    243354,243368,243382,243396,243409,243423,243437,243451,243465,243478,
    243492,243506,243520,243534,243547,243561,243575,243589,243603,243616,
    243630,243644,243658,243672,243686,243699,243713,243727,243741,243755,
    243769,243782,243796,243810,243824,243838,243852,243865,243879,243893,
    243907,243921,243935,243949,243962,243976,243990,244004,244018,244032,
    244046,244059,244073,244087,244101
};

// Constant-time conditional subtraction
static inline uint32_t cond_sub(uint32_t r, uint32_t n) {
    uint32_t mask = 0 - (r >> 31);
    return r + (n & mask);
}

// Barrett reduction: a mod (N − i)
static inline uint32_t reduce_barrett(uint32_t a, size_t i) {
    uint32_t q = ((uint64_t)a * m_val[i]) >> 32;
    uint32_t n = (uint32_t)(PARAM_N - i);
    uint32_t r = a - q * n;
    return cond_sub(r, n);
}

// MulShift reduction: floor(a * (N − i) / 2^32)
static inline uint32_t reduce_mulshift(uint32_t a, size_t i) {
    uint32_t n = (uint32_t)(PARAM_N - i);
    return ((uint64_t)a * n) >> 32;
}

// Selector
static volatile int use_barrett = 1;
void hqc128_use_barrett(void)  { use_barrett = 1; }
void hqc128_use_mulshift(void) { use_barrett = 0; }

// Dispatch
static inline uint32_t reduce_mod(uint32_t a, size_t i) {
    return use_barrett ? reduce_barrett(a, i)
                       : reduce_mulshift(a, i);
}

//------------------------------------------------------------------------------
// Bias-test harness

#define MULT 10000

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <barrett|mulshift>\n", argv[0]);
        return 1;
    }
    if (strcmp(argv[1], "barrett") == 0) {
        hqc128_use_barrett();
    } else if (strcmp(argv[1], "mulshift") == 0) {
        hqc128_use_mulshift();
    } else {
        fprintf(stderr, "Unknown reducer '%s'\n", argv[1]);
        return 1;
    }

    const uint32_t N = PARAM_N;
    size_t samples = (size_t)N * MULT;
    uint32_t *hist = calloc(N, sizeof(uint32_t));
    if (!hist) { perror("calloc"); return 1; }

    srand((unsigned)time(NULL));
    for (size_t k = 0; k < samples; k++) {
        uint32_t a = ((uint32_t)rand() << 16) ^ (uint32_t)rand();
        size_t i = k % 75;                // cycle through all i = 0..74
        uint32_t out = reduce_mod(a, i);
        uint32_t n = PARAM_N - (uint32_t)i;
        if (out >= n) out %= n;           // clamp into [0, n)
        hist[out]++;
    }

    double expected = (double)samples / N;
    double max_dev = 0, sum_dev = 0;
    for (uint32_t v = 0; v < N; v++) {
        double d = fabs(hist[v] - expected);
        sum_dev += d;
        if (d > max_dev) max_dev = d;
    }
    double mean_dev = sum_dev / N;

    printf("Bias test for HQC-128 (%s)\n", argv[1]);
    printf("  N             = %u\n", N);
    printf("  total samples = %zu\n", samples);
    printf("  Expected/bin  = %.2f\n", expected);
    printf("  Max dev       = %.2f (%.5f%%)\n",
           max_dev, 100.0 * max_dev / expected);
    printf("  Mean dev      = %.2f (%.5f%%)\n",
           mean_dev, 100.0 * mean_dev / expected);

    free(hist);
    return 0;
}
