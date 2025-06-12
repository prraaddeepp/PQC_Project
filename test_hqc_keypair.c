#include <stdio.h>
#include <stdlib.h>
#include <oqs/kem.h>


int main() {
   OQS_KEM *kem = OQS_KEM_new(OQS_KEM_alg_hqc_128);
   if (kem == NULL) {
       printf("Failed to initialize HQC KEM\n");
       return 1;
   }


   uint8_t *public_key = malloc(kem->length_public_key);
   uint8_t *secret_key = malloc(kem->length_secret_key);


   if (OQS_KEM_keypair(kem, public_key, secret_key) != OQS_SUCCESS) {
       printf("Key generation failed\n");
       return 1;
   }


   printf("Key generation successful (C test)!\n");


   free(public_key);
   free(secret_key);
   OQS_KEM_free(kem);


   return 0;
}
