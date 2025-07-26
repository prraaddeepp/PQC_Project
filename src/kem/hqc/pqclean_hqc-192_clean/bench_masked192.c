// full_sampler_bench.c
/*****************************************************************
 * HQC‑192 Vector Sampling Benchmark
 *   • Unmasked (Barrett) via PQClean
 *   • Masked Boolean‑Barrett (Alg 11′)
 *   • Masked Boolean‑MulShift (Alg 12′)
 *
 * Reports: min / avg / max / stddev cycles/sample (ITERS per run)
 *****************************************************************/
 #include <stdio.h>
 #include <stdlib.h>
 #include <stdint.h>
 #include <math.h>
 #include <assert.h>
 #include <string.h>
 #include <x86intrin.h>
 
 #include "shake_prng.h"    // PQCLEAN_HQC192_CLEAN_seedexpander_*
 #include "vector.h"        // single_bit_mask(), compare_u32(), m_val[], PQCLEAN_HQC192_CLEAN_vect_set_random_fixed_weight()
 #include "parameters.h"    // PARAM_N, PARAM_OMEGA_R, VEC_N_SIZE_64
 #include "hqc_reducer.h"   // hqc192_use_barrett(), hqc192_use_mulshift()
 
 // bench parameters
 #define VEC_W    PARAM_OMEGA_R   
 #define ITERS    50000          // samples per timed run
 #define RUNS     100
 #define WARMUP   2
 #define BARRIER() asm volatile("" ::: "memory")
 
 // sampler_fn signature
 typedef void (*sampler_fn)(uint32_t seed,
                            const uint32_t *rnds,
                            uint64_t out[VEC_N_SIZE_64]);
 
 // 1) Unmasked HQC sampler (Barrett mode)
 static void sample_hqc_unmasked(uint32_t       seed,
                                 const uint32_t *rnds,
                                 uint64_t       out[VEC_N_SIZE_64])
 {
     (void)rnds;
     seedexpander_state ctx;
     PQCLEAN_HQC192_CLEAN_seedexpander_init(&ctx, (uint8_t*)&seed, 4);
     PQCLEAN_HQC192_CLEAN_vect_set_random_fixed_weight(&ctx, out, VEC_W);
     PQCLEAN_HQC192_CLEAN_seedexpander_release(&ctx);
 }
 
 // 2) First‑order Boolean‑masking gadgets
 typedef struct { uint32_t s[2]; } masked_u32;
 typedef struct { uint32_t s[2]; } masked_u32_arith;
 static inline void sec_mul_const(const masked_u32 *x, uint32_t c, masked_u32 *z) {
     uint64_t p0 = (uint64_t)x->s[0]*c;
     uint64_t p1 = (uint64_t)x->s[1]*c;
     z->s[0] = (uint32_t)(p0>>32);
     z->s[1] = (uint32_t)(p1>>32);
 }
 static inline void sec_sub(const masked_u32 *x, const masked_u32 *y, masked_u32 *z) {
     z->s[0]=x->s[0]-y->s[0]; z->s[1]=x->s[1]-y->s[1];
 }
 static inline void sec_add_const(const masked_u32 *x, uint32_t c, masked_u32 *z) {
     z->s[0]=x->s[0]+c; z->s[1]=x->s[1];
 }
 static inline void refresh(masked_u32 *x, uint32_t rnd) {
     x->s[0]^=rnd; x->s[1]^=rnd;
 }
 static inline uint32_t sec_equals_u32(const masked_u32 *x, uint32_t v) {
     return (((x->s[0]^x->s[1])==v)?0xFFFFFFFFu:0u);
 }
 static inline void sec_if(const masked_u32 *x, const masked_u32 *y,
                           uint32_t cond, masked_u32 *z) {
     z->s[0]=(x->s[0]&cond)|(y->s[0]&~cond);
     z->s[1]=(x->s[1]&cond)|(y->s[1]&~cond);
 }
 
 // 3) Masked Barrett sampler (Alg 11′)
 static void sample_hqc_masked_barrett(uint32_t seed,
                                       const uint32_t *rnds,
                                       uint64_t out[VEC_N_SIZE_64])
 {
     seedexpander_state ctx;
     PQCLEAN_HQC192_CLEAN_seedexpander_init(&ctx,(uint8_t*)&seed,4);
     uint32_t support[VEC_W];
     for(int i=0;i<VEC_W;i++){
         uint8_t buf[4];
         PQCLEAN_HQC192_CLEAN_seedexpander(&ctx,buf,4);
         uint32_t a=(uint32_t)buf[0]|(uint32_t)buf[1]<<8|
                    (uint32_t)buf[2]<<16|(uint32_t)buf[3]<<24;
         uint32_t r0 = rnds[0], r1 = rnds[1];
         masked_u32 A={.s={a, r0}}, Q, QN, AQ, A_, R;
         sec_mul_const(&A,m_val[i],&Q);
         uint32_t n=PARAM_N-(uint32_t)i;
         sec_mul_const(&Q,n,&QN);
         refresh(&A, r0);
         sec_sub(&A,&QN,&AQ);
         sec_add_const(&AQ,(uint32_t)0-n,&A_);
         masked_u32 Z={.s={A_.s[0]>>31, A_.s[1]>>31}};
         uint32_t cond=sec_equals_u32(&Z,1);
         refresh(&AQ, r1);
         sec_if(&AQ,&A_,cond,&R);
         support[i] = (uint32_t)((i + (R.s[0] ^ R.s[1])) % PARAM_N);
     }
     PQCLEAN_HQC192_CLEAN_seedexpander_release(&ctx);
     for(int i=VEC_W-1;i>=0;i--){
         uint32_t found=0;
         for(int j=i+1;j<VEC_W;j++){
             uint32_t diff=support[j]^support[i];
             found|=(1-((diff|-diff)>>31));
         }
         uint32_t m=0-found;
         support[i]=(m&(uint32_t)i)|(~m&support[i]);
     }
     memset(out,0,VEC_N_SIZE_64*sizeof*out);
     for(int i=0;i<VEC_W;i++){
         uint32_t w=support[i]>>6, b=support[i]&0x3F;
         out[w]|=single_bit_mask(b);
     }
 }
 
 // 4) Masked MulShift sampler (Alg 12′)
 static void sample_hqc_masked_mulshift(uint32_t seed,
                                        const uint32_t *rnds,
                                        uint64_t out[VEC_N_SIZE_64])
 {
     seedexpander_state ctx;
     PQCLEAN_HQC192_CLEAN_seedexpander_init(&ctx,(uint8_t*)&seed,4);
     uint32_t support[VEC_W];
     for(int i=0;i<VEC_W;i++){
         uint8_t buf[4];
         PQCLEAN_HQC192_CLEAN_seedexpander(&ctx,buf,4);
         uint32_t a=(uint32_t)buf[0]|(uint32_t)buf[1]<<8|
                    (uint32_t)buf[2]<<16|(uint32_t)buf[3]<<24;
         uint32_t r0 = rnds[0], r1 = rnds[1];
         masked_u32 A={.s={a, r0}}, R;
         uint32_t n=PARAM_N-(uint32_t)i;
         sec_mul_const(&A,n,&R);
         refresh(&R, r1);
         support[i] = (uint32_t)((i + (R.s[0] ^ R.s[1])) % PARAM_N);
     }
     PQCLEAN_HQC192_CLEAN_seedexpander_release(&ctx);
     for(int i=VEC_W-1;i>=0;i--){
         uint32_t found=0;
         for(int j=i+1;j<VEC_W;j++){
             uint32_t diff=support[j]^support[i];
             found|=(1-((diff|-diff)>>31));
         }
         uint32_t m=0-found;
         support[i]=(m&(uint32_t)i)|(~m&support[i]);
     }
     // --- Hamming weight check ---
     memset(out,0,VEC_N_SIZE_64*sizeof*out);
     for(int i=0;i<VEC_W;i++){
         uint32_t w=support[i]>>6, b=support[i]&0x3F;
         out[w]|=single_bit_mask(b);
     }
 }
 // Secure arithmetic multiplication by public c:
 // share‑wise multiply, no shift needed
 static inline void sec_mul_const_arith(const masked_u32_arith *x,
    uint32_t c,
    masked_u32_arith *z) {
    // each share multiplied by c
    z->s[0] = x->s[0] * c;
    z->s[1] = x->s[1] * c;
 }

 // Re‐randomize arithmetic shares: (x0,x1) -> (x0+r, x1–r)
 static inline void arithmetic_refresh(masked_u32_arith *x, uint32_t rnd) {
 x->s[0] += rnd;
 x->s[1] -= rnd;
 }

// Boolean→Arithmetic conversion (BtoA):
// Given ⟨a⟩_bool where a = a0⊕a1, produce ⟨a⟩_arith with sum = a
 static inline masked_u32_arith BtoA(const masked_u32 *x_bool,
 uint32_t rnd) {
    masked_u32_arith y;
    uint32_t a = x_bool->s[0] ^ x_bool->s[1];
    y.s[0] = rnd;            // random share
    y.s[1] = a - rnd;        // ensures y0+y1 = a mod 2^32
    return y;
 }

 // Arithmetic→Boolean conversion (AtoB):
 // Given ⟨a⟩_arith where a=y0+y1, produce ⟨a⟩_bool with xor = a
 static inline masked_u32 AtoB(const masked_u32_arith *x_arith,
 uint32_t rnd) {
    masked_u32 z;
    uint32_t a = x_arith->s[0] + x_arith->s[1];
    z.s[0] = rnd;            // random share
    z.s[1] = a ^ rnd;        // ensures z0⊕z1 = a
    return z;
 }

 ////////////////////////////////////////////////////////////////////////////////
 // 4) Masked Arithmetic MulShift sampler (Alg 12′‐style for MulShift)
 static void sample_hqc_masked_mulshift_arith(uint32_t seed,
          const uint32_t *rnds,
          uint64_t out[VEC_N_SIZE_64])
 {
    seedexpander_state ctx;
    PQCLEAN_HQC192_CLEAN_seedexpander_init(&ctx, (uint8_t*)&seed, 4);
    uint32_t support[VEC_W];

    for(int i = 0; i < VEC_W; i++){
        // 0) grab 32‑bit candidate a in Boolean shares:
        uint8_t buf[4];
        PQCLEAN_HQC192_CLEAN_seedexpander(&ctx, buf, 4);
        uint32_t a = (uint32_t)buf[0]
        | (uint32_t)buf[1]<<8
        | (uint32_t)buf[2]<<16
        | (uint32_t)buf[3]<<24;
        
        uint32_t r0 = rnds[0], r1 = rnds[1], r2 = rnds[2];
        masked_u32   A_bool  = { .s = { a, r0 } };
        masked_u32_arith A_arith = BtoA(&A_bool, r1);
        // --- 2) 64-bit multiplication + extract high word ---
        uint32_t n = PARAM_N - i;  // Safe for HQC-128 (PARAM_N=17669)
        uint64_t p0 = (uint64_t)A_arith.s[0] * n;
        uint64_t p1 = (uint64_t)A_arith.s[1] * n;
        masked_u32_arith R_arith = {
            .s = {(uint32_t)(p0 >> 32), (uint32_t)(p1 >> 32)}
        };

        // 3) Arithmetic→Boolean using r2
        masked_u32 R_bool = AtoB(&R_arith, r2);

        // --- 3) Arithmetic→Boolean ---
        uint32_t idx = (i + (R_bool.s[0] ^ R_bool.s[1]))     % PARAM_N;
        support[i] = idx;
    }

    PQCLEAN_HQC192_CLEAN_seedexpander_release(&ctx);

    // --- Deduplication (optimized) ---
    for (int i = VEC_W - 1; i >= 0; i--) {
        uint32_t found = 0;
        for (int j = i + 1; j < VEC_W; j++) {
            found |= (support[j] == support[i]) ? 0xFFFFFFFFu : 0u;
        }
        support[i] = (found & i) | (~found & support[i]);
    }

    // --- Pack into bitmask ---
    memset(out, 0, VEC_N_SIZE_64 * sizeof *out);
    for (int i = 0; i < VEC_W; i++) {
        out[support[i] >> 6] |= single_bit_mask(support[i] & 0x3F);
    }
 }
 
 // 5) Benchmark driver
 static void bench(const char *label, sampler_fn sampler){
     uint64_t times[RUNS];
     uint32_t *seeds, *rnds;
     if (posix_memalign((void**)&seeds, 64, ITERS * sizeof *seeds) != 0 ||
        posix_memalign((void**)&rnds,  64, 3*ITERS * sizeof *rnds) != 0) {
        perror("posix_memalign");
        exit(1);
     }
     assert(seeds&&rnds);
     for(int i=0;i<ITERS;i++){
         seeds[i]=0x9E3779B9u*i+0x7F4A7C15u;
         uint32_t base = 0xC6BC2796u * i + 0x2545F491u;
         rnds[3*i + 0] = base;
         rnds[3*i + 1] = base ^ 0xDEADBEEF;
         rnds[3*i + 2] = base + 0x9E3779B9u;
     }
     uint64_t dummy[VEC_N_SIZE_64];
     for(int w=0;w<WARMUP;w++)
         for(int i=0;i<ITERS;i++) sampler(seeds[i],rnds+3*i,dummy);
     for(int r=0;r<RUNS;r++){
         unsigned junk; BARRIER();
         uint64_t t0=__rdtscp(&junk);
         for(int i=0;i<ITERS;i++) sampler(seeds[i],rnds+3*i,dummy);
         BARRIER();
         uint64_t t1=__rdtscp(&junk);
         times[r]=t1-t0;
     }
     double sum=0,sum2=0,mn=1e18,mx=0;
     for(int r=0;r<RUNS;r++){
         double c=(double)times[r]/ITERS;
         sum+=c; sum2+=c*c;
         if(c<mn) mn=c; if(c>mx) mx=c;
     }
     double avg=sum/RUNS, std=sqrt(sum2/RUNS-avg*avg);
     printf("%-24s | avg%7.2f | min%7.2f | max%7.2f | std%5.2f\n",
            label,avg,mn,mx,std);
     free(seeds); free(rnds);
 }
 
 int main(void){
     puts("\nHQC‑192 Vector Sampling (ω=114) Benchmark\n");
     bench("HQC masked MulShift Arithmetic",       sample_hqc_masked_mulshift_arith);
     bench("HQC masked Barrett", sample_hqc_masked_barrett);
     bench("HQC masked MulShift",sample_hqc_masked_mulshift);
     return 0;
 }
 