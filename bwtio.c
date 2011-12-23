#include <mpi.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "bwt.h"
#include "utils.h"
#include <math.h>

void bwt_dump_bwt(const char *fn, const bwt_t *bwt)
{
	FILE *fp;
	fp = xopen(fn, "wb");
	fwrite(&bwt->primary, sizeof(bwtint_t), 1, fp);
	fwrite(bwt->L2+1, sizeof(bwtint_t), 4, fp);
	fwrite(bwt->bwt, sizeof(bwtint_t), bwt->bwt_size, fp);
	fclose(fp);
}

void bwt_dump_sa(const char *fn, const bwt_t *bwt)
{
	FILE *fp;
	fp = xopen(fn, "wb");
	fwrite(&bwt->primary, sizeof(bwtint_t), 1, fp);
	fwrite(bwt->L2+1, sizeof(bwtint_t), 4, fp);
	fwrite(&bwt->sa_intv, sizeof(bwtint_t), 1, fp);
	fwrite(&bwt->seq_len, sizeof(bwtint_t), 1, fp);
	fwrite(bwt->sa + 1, sizeof(bwtint_t), bwt->n_sa - 1, fp);
	fclose(fp);
}

void bwt_restore_sa(const char *fn, bwt_t *bwt)
{
	char skipped[256];
	FILE *fp;
	bwtint_t primary;
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);	

	fp = xopen(fn, "rb");
	fread(&primary, sizeof(bwtint_t), 1, fp);
	xassert(primary == bwt->primary, "SA-BWT inconsistency: primary is not the same.");
	fread(skipped, sizeof(bwtint_t), 4, fp); // skip
	fread(&bwt->sa_intv, sizeof(bwtint_t), 1, fp);
	fread(&primary, sizeof(bwtint_t), 1, fp);
	xassert(primary == bwt->seq_len, "SA-BWT inconsistency: seq_len is not the same.");

	bwt->n_sa = (bwt->seq_len + bwt->sa_intv) / bwt->sa_intv;
	bwt->sa = (bwtint_t*)calloc(bwt->n_sa, sizeof(bwtint_t));
	bwt->sa[0] = -1;
	/*all of our processors can do individual reads up until this point because
	**the reads have been small.  This next step is reading the entire Suffix Array.
	**Processor 0 will read it and broadcast it		*/
	if (rank != 0) {
	
		fclose(fp);
	} else {
		
		fprintf(stderr, "Broadcasting SA... ");
		fread(bwt->sa + 1, sizeof(bwtint_t), bwt->n_sa - 1, fp);
		fclose(fp);
	}
	MPI_Bcast(bwt->sa + 1, bwt->n_sa - 1, MPI_INT, 0, MPI_COMM_WORLD);	
	
	if (rank == 0) {
	
		fprintf(stderr, "done!\n");
	}
}

bwt_t *bwt_restore_bwt(const char *fn)
{
	bwt_t *bwt;
	FILE *fp;

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);	
	
	
		bwt = (bwt_t*)calloc(1, sizeof(bwt_t));
	
	//processor 0 will read in the BWT information and broadcast it
	if (rank == 0) {
		
		fprintf(stderr, "Broadcasting BWT (this may take a while)... ");		
		fp = xopen(fn, "rb");
		fseek(fp, 0, SEEK_END);
		bwt->bwt_size = (ftell(fp) - sizeof(bwtint_t) * 5) >> 2;
		bwt->bwt = (uint32_t*)calloc(bwt->bwt_size, 4);
		fseek(fp, 0, SEEK_SET);
		fread(&bwt->primary, sizeof(bwtint_t), 1, fp);
		fread(bwt->L2+1, sizeof(bwtint_t), 4, fp);
		fread(bwt->bwt, 4, bwt->bwt_size, fp);
		bwt->seq_len = bwt->L2[4];
		fclose(fp);
	}
	
	MPI_Bcast(&bwt->bwt_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
	if (rank != 0) bwt->bwt = (uint32_t*)calloc(bwt->bwt_size, 4);
	MPI_Bcast(bwt->bwt, bwt->bwt_size, MPI_INT, 0, MPI_COMM_WORLD);
	
	MPI_Bcast(&bwt->primary, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(bwt->L2+1, 4, MPI_INT, 0, MPI_COMM_WORLD);
	if (rank != 0) {
		
		bwt->seq_len = bwt->L2[4];
	} else {

		fprintf(stderr, "done!\n");
	}
	
	bwt_gen_cnt_table(bwt);	
	return bwt;
}

void bwt_destroy(bwt_t *bwt)
{
	if (bwt == 0) return;
	free(bwt->sa); free(bwt->bwt);
	free(bwt);
}
