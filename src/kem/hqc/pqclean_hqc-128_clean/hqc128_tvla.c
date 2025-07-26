// hqc128_tvla.c
// Test‑Vector Leakage Assessment for HQC‑128 masked samplers

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <x86intrin.h>         // __rdtscp
#include "shake_prng.h"        // PQCLEAN_HQC128_CLEAN_seedexpander_*
#include "vector.h"            // single_bit_mask(), m_val[], PQCLEAN_HQC128_CLEAN_vect_set_random_fixed_weight()
#include "parameters.h"        // PARAM_N, PARAM_OMEGA_R, VEC_N_SIZE_64
#include "randombytes.h" 
// -----------------------------------------------------------------------------
// Config
// -----------------------------------------------------------------------------
#define VEC_W      PARAM_OMEGA_R    // 114
#define ITERS      50000            // samples per batch
#define BATCHES    100              // number of batches
#define THRESHOLD  4.5              // |t| > 4.5 ⇒ leakage

#define BARRIER()  asm volatile("" ::: "memory")

// -----------------------------------------------------------------------------
// First‐order Boolean masking helpers (exactly as in your bench code)
// -----------------------------------------------------------------------------
typedef struct { uint32_t s[2]; } masked_u32;

static inline void sec_mul_const(const masked_u32 *x, uint32_t c, masked_u32 *z) {
    uint64_t p0 = (uint64_t)x->s[0]*c, p1 = (uint64_t)x->s[1]*c;
    z->s[0] = (uint32_t)(p0>>32);
    z->s[1] = (uint32_t)(p1>>32);
}
static inline void refresh(masked_u32 *x, uint32_t rnd) {
    x->s[0] ^= rnd; x->s[1] ^= rnd;
}

// -----------------------------------------------------------------------------
//  Masked Boolean‑MulShift sampler (Alg 12′)
// -----------------------------------------------------------------------------
static void sample_hqc_masked_mulshift(uint32_t seed,
                                       const uint32_t *rnds,
                                       uint64_t out[VEC_N_SIZE_64])
{
    seedexpander_state ctx;
    PQCLEAN_HQC128_CLEAN_seedexpander_init(&ctx,(uint8_t*)&seed,4);
    uint32_t support[VEC_W];
    for(int i=0;i<VEC_W;i++){
        uint8_t buf[4];
        PQCLEAN_HQC128_CLEAN_seedexpander(&ctx,buf,4);
        uint32_t a = (uint32_t)buf[0]
                   | (uint32_t)buf[1]<<8
                   | (uint32_t)buf[2]<<16
                   | (uint32_t)buf[3]<<24;
        uint32_t r0 = rnds[0], r1 = rnds[1];
        masked_u32 A = { .s = { a, r0 } }, R;
        uint32_t n = PARAM_N - i;
        sec_mul_const(&A, n, &R);
        refresh(&R, r1);
        support[i] = (uint32_t)((i + (R.s[0] ^ R.s[1])) % PARAM_N);
    }
    PQCLEAN_HQC128_CLEAN_seedexpander_release(&ctx);

    // deduplicate
    for(int i=VEC_W-1;i>=0;i--){
        uint32_t found=0;
        for(int j=i+1;j<VEC_W;j++){
            uint32_t diff = support[j]^support[i];
            found |= (1u-((diff| -diff)>>31));
        }
        uint32_t m = 0-found;
        support[i] = (m & (uint32_t)i) | (~m & support[i]);
    }

    // pack into bitmask vector
    memset(out,0,sizeof(uint64_t)*VEC_N_SIZE_64);
    for(int i=0;i<VEC_W;i++){
        uint32_t w = support[i]>>6, b = support[i]&0x3F;
        out[w] |= single_bit_mask(b);
    }
}

// -----------------------------------------------------------------------------
// Welford’s online mean/variance (for each class)
// -----------------------------------------------------------------------------
typedef struct {
    double mean, m2;
    size_t  n;
} stat_t;

static void stat_init(stat_t *s){
    s->mean = 0;
    s->m2   = 0;
    s->n    = 0;
}

static void stat_update(stat_t *s, double x){
    s->n++;
    double d = x - s->mean;
    s->mean += d / s->n;
    s->m2   += d * (x - s->mean);
}

static double stat_variance(const stat_t *s){
    return (s->n > 1) ? (s->m2/(s->n-1)) : 0.0;
}

// -----------------------------------------------------------------------------
// Main: fixed‑vs‑random TVLA
// -----------------------------------------------------------------------------
int main(void){
    uint32_t seeds[ITERS], rnds[3*ITERS];
    uint64_t outbuf[VEC_N_SIZE_64];

    // Pre‐compute seeds
    for(int i=0;i<ITERS;i++){
        seeds[i] = 0x9E3779B9u*i + 0x7F4A7C15u;
    }

    // Batches of ITERS samples
    for(int batch=0; batch<BATCHES; batch++){
        stat_t s_fixed, s_rand;
        stat_init(&s_fixed);
        stat_init(&s_rand);

        // prepare randomness for this batch: interleaved fixed(0) / random(1)
        for(int i=0;i<ITERS;i++){
            if((i&1)==0){
                // fixed class: deterministic but non‐trivial
                uint32_t base = 0xC6BC2796u * i + 0x2545F491u;
                rnds[3*i+0] = base;
                rnds[3*i+1] = base ^ 0xDEADBEEF;
                rnds[3*i+2] = base + 0x9E3779B9u;
            } else {
                // random class: true randomness
                randombytes((uint8_t*)&rnds[3*i], 3*sizeof(uint32_t));
            }
        }

        // Warmup (cold caches)
        for(int w=0; w<1000; w++){
            sample_hqc_masked_mulshift(seeds[w%ITERS], &rnds[3*(w%ITERS)], outbuf);
        }

        // Measure
        for(int i=0;i<ITERS;i++){
            unsigned junk;
            BARRIER();
            uint64_t t0 = __rdtscp(&junk);
            BARRIER();
            sample_hqc_masked_mulshift(seeds[i], &rnds[3*i], outbuf);
            BARRIER();
            uint64_t t1 = __rdtscp(&junk);
            BARRIER();

            double cycles = (double)(t1 - t0);
            if((i&1)==0) stat_update(&s_fixed, cycles);
            else         stat_update(&s_rand,   cycles);
        }

        // Welch’s t‑statistic
        double m0 = s_fixed.mean,  v0 = stat_variance(&s_fixed), n0 = s_fixed.n;
        double m1 = s_rand.mean,   v1 = stat_variance(&s_rand),  n1 = s_rand.n;
        double num = m0 - m1;
        double den = sqrt(v0/n0 + v1/n1);
        double tstat = num / den;

        printf("Batch %3d: t = %+6.3f\n", batch, tstat);
    }

    return 0;
}
