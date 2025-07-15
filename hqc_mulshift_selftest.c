#include <oqs/kem.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void die(const char *msg) {
    fprintf(stderr, "❌ %s\n", msg);
    exit(EXIT_FAILURE);
}

int main(int argc, char **argv)
{
    const size_t ITER = (argc >= 2) ? strtoull(argv[1], NULL, 10) : 10000;

    /* --- 1. init --- */
    OQS_KEM *kem = OQS_KEM_new(OQS_KEM_alg_hqc_256);
    if (!kem) die("OQS_KEM_new failed");

    uint8_t *pk  = malloc(kem->length_public_key);
    uint8_t *sk  = malloc(kem->length_secret_key);
    uint8_t *ct  = malloc(kem->length_ciphertext);
    uint8_t *ss1 = malloc(kem->length_shared_secret);
    uint8_t *ss2 = malloc(kem->length_shared_secret);
    if (!pk || !sk || !ct || !ss1 || !ss2) die("malloc");


    /* --- 3. main loop --- */
    for (size_t i = 0; i < ITER; i++) {
        if (OQS_KEM_keypair(kem, pk, sk) != OQS_SUCCESS) die("keypair fail");
        if (OQS_KEM_encaps(kem, ct, ss1, pk) != OQS_SUCCESS) die("encap fail");
        if (OQS_KEM_decaps(kem, ss2, ct, sk) != OQS_SUCCESS) die("decap fail");

        if (memcmp(ss1, ss2, kem->length_shared_secret) != 0) {
            fprintf(stderr, "mismatch at iteration %zu\n", i);
            die("shared secrets differ – reducer is wrong");
        }

        /* ↓↓↓ progress message ↓↓↓ */
        printf("pass %zu/%zu\r", i + 1, ITER);   /* overwrite same line */
        fflush(stdout);                           /* ensure it appears */
    }
    putchar('\n');   /* move to new line before final ✅ */


    /* --- 4. clean up --- */
    free(pk); free(sk); free(ct); free(ss1); free(ss2);
    OQS_KEM_free(kem);
    return EXIT_SUCCESS;
}
