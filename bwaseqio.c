#include <mpi.h>
#include <time.h>
#include <stdio.h>
#include <zlib.h>
#include <ctype.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>

#include "bwtaln.h"
#include "utils.h"
#include "bamlite.h"

#include "kseq.h"
//here we modify kseq - from compressed to uncompressed files
KSEQ_INIT(FILE*, fread)

extern unsigned char nst_nt4_table[256];
static char bam_nt16_nt4_table[] = { 4, 0, 1, 4, 2, 4, 4, 4, 3, 4, 4, 4, 4, 4, 4, 4 };

struct __bwa_seqio_t {
	// for BAM input
	int is_bam, which; // 1st bit: read1, 2nd bit: read2, 3rd: SE
	bamFile fp;
	// for fastq input
	kseq_t *ks;
	long long int pos, endPos;
};

long long int getPos(bwa_seqio_t *bs) {
       return ftello64(bs->ks->f->f) - bs->ks->f->end + bs->ks->f->begin;
}

void setPos(bwa_seqio_t *bs, long long int pos) {
       bs->pos = pos;
       fseeko64(bs->ks->f->f, bs->pos, SEEK_SET);
       ks_rewind(bs->ks->f);
}

char *trimReadName (char *name) { // trim /[12]$
       int t = strlen(name);
       if (t > 2 && name[t-2] == '/' && (name[t-1] == '1' || name[t-1] == '2')) name[t-2] = '\0';
    return name;
}

long long int fsize(const char *fn) {
    struct stat st;

    if (stat(fn, &st) == 0)
        return st.st_size;

    return -1;
}


void sharePositions (bwa_seqio_t *bs) {
       int rank, size;
       MPI_Comm_rank(MPI_COMM_WORLD, &rank);
       MPI_Comm_size(MPI_COMM_WORLD, &size);

       // communicate with neighbor ranks to determine end positions
       MPI_Request request;
       if (rank != 0) {
               MPI_Isend(&bs->pos, 1, MPI_LONG_LONG_INT, rank-1, 0, MPI_COMM_WORLD, &request);
       }
       if (rank != (size-1)) {
               MPI_Recv(&bs->endPos, 1, MPI_LONG_LONG_INT, rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
       }
    if (rank != 0)
       MPI_Wait(&request, MPI_STATUS_IGNORE);
}

void skipToNextRecord(bwa_seqio_t *bs) {
       kstring_t buf;
       buf.l = buf.m = 0;
       buf.s = NULL;
       int dret;
       char c;
       long long int possiblePos = -1;
       while (possiblePos < 0) {
               ks_getuntil(bs->ks->f, '\n', &buf, &dret);
               c = ks_getc(bs->ks->f);
               if (c == '>' || c == '@') {
                       possiblePos = getPos(bs) -1;
                       // > and @ are valid quality characters, check the next line for header
                       ks_getuntil(bs->ks->f, '\n', &buf, &dret);
                       c = ks_getc(bs->ks->f);
                       if (c == '>' || c == '@')
                               possiblePos = getPos(bs) -1;
               }
       }
       setPos(bs, possiblePos);
       if (buf.s)
               free(buf.s);
}

void skipToNextPairedRecord(bwa_seqio_t *bs1, bwa_seqio_t *bs2) {
       int rank;
       char *read1Start = NULL, *read2Start = NULL;
       long long int pos1Start = getPos(bs1), pos2Start = getPos(bs2);
       long long int pos1 = pos1Start, pos2 = pos2Start;
       kseq_t *seq1 = bs1->ks, *seq2 = bs2->ks;
       int l1,l2;
       int round = 0;
       MPI_Comm_rank(MPI_COMM_WORLD, &rank);
       int found = (rank == 0) ? 1 : 0;
       while (found == 0 && round++ < NUM_NEEDED*2) {
               pos1 = getPos(bs1);
               l1 = kseq_read(seq1);
               pos2 = getPos(bs2);
               l2 = kseq_read(seq2);
               if (l1 <= 0 || l2 <= 0)
                       break;

               if (round == 1) {
                       read1Start = trimReadName(strdup(seq1->name.s));
                       read2Start = trimReadName(strdup(seq2->name.s));
               }
               if (strcmp(read1Start, trimReadName(seq2->name.s)) == 0) {
                       fprintf (stderr, "Proc %d: [skipToNextPairedRecord] found %s %s\n", rank, read1Start, seq2->name.s);
                       setPos(bs1, pos1Start);
                       setPos(bs2, pos2);
                       found = 1;
                       break;
               }
               if (strcmp(read2Start, trimReadName(seq1->name.s)) == 0) {
                       fprintf (stderr, "Proc %d: [skipToNextPairedRecord] found %s %s\n", rank, read2Start, seq1->name.s);
                       setPos(bs1, pos1);
                       setPos(bs2, pos2Start);
                       found = 1;
                       break;
               }
       };
       if (found != 1) {
               fprintf(stderr, "Proc %d: Could not find a matching pair in the two files! %d %s %s\n", rank, round, read1Start, read2Start);
               exit(1);
       }
       sharePositions(bs1);
       sharePositions(bs2);
       if (read1Start)
               free(read1Start);
       if (read2Start)
               free(read2Start);
}

void mergeFilesIntoOne(char *myFileName, char *ourFileName) {
	void *fp;
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	double time = MPI_Wtime();
	fprintf(stderr, "Proc %d: [mergeFilesIntoOne] Starting merge of %s into %s\n", rank, myFileName, ourFileName);
	int fd = open(myFileName, O_RDONLY);
	int fileSize = fsize(myFileName);
	fp = mmap(NULL, fileSize, PROT_READ, MAP_SHARED, fd, 0);
	MPI_File ourFile;
	MPI_File_open(MPI_COMM_WORLD, ourFileName, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &ourFile);
	MPI_File_write_ordered(ourFile, fp, fileSize, MPI_BYTE, MPI_STATUS_IGNORE);
	munmap(fp, fileSize);
	close(fd);
	unlink(myFileName);
	MPI_File_close(&ourFile);
	time = MPI_Wtime() - time;
	fprintf(stderr, "Proc %d: [mergeFilesIntoOne] Finished merge in %0.2lf secs\n", rank,time);
}

bwa_seqio_t *bwa_bam_open(const char *fn, int which)
{
	bwa_seqio_t *bs;
	bam_header_t *h;
	bs = (bwa_seqio_t*)calloc(1, sizeof(bwa_seqio_t));
	bs->is_bam = 1;
	bs->which = which;
	bs->fp = bam_open(fn, "r");
	h = bam_header_read(bs->fp);
	bam_header_destroy(h);
	return bs;
}

bwa_seqio_t *bwa_seq_open(const char *fn)
{
	FILE * fp;
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	bwa_seqio_t *bs;
	bs = (bwa_seqio_t*)calloc(1, sizeof(bwa_seqio_t));
	fp = xopen(fn, "r");
	bs->ks = kseq_init(fp);
	long long int fileSize = fsize(fn);
	setPos(bs, rank * (fileSize/size));
	bs->endPos = (rank == (size-1)) ? fsize(fn) : (rank+1) * (fileSize/size);

	if (rank != 0)
		skipToNextRecord(bs);

	fprintf(stderr, "Proc %d: [bwa_seq_open] seeked to %lld in %s\n", rank, bs->pos, fn);

	sharePositions(bs);
	return bs;

}

void bwa_seq_close(bwa_seqio_t *bs)
{
	if (bs == 0) return;
	if (bs->is_bam) bam_close(bs->fp);
	else {
		fclose(bs->ks->f->f);
		kseq_destroy(bs->ks);
	}
	free(bs);
}

void seq_reverse(int len, ubyte_t *seq, int is_comp)
{
	int i;
	if (is_comp) {
		for (i = 0; i < len>>1; ++i) {
			char tmp = seq[len-1-i];
			if (tmp < 4) tmp = 3 - tmp;
			seq[len-1-i] = (seq[i] >= 4)? seq[i] : 3 - seq[i];
			seq[i] = tmp;
		}
		if (len&1) seq[i] = (seq[i] >= 4)? seq[i] : 3 - seq[i];
	} else {
		for (i = 0; i < len>>1; ++i) {
			char tmp = seq[len-1-i];
			seq[len-1-i] = seq[i]; seq[i] = tmp;
		}
	}
}

int bwa_trim_read(int trim_qual, bwa_seq_t *p)
{
	int s = 0, l, max = 0, max_l = p->len - 1;
	if (trim_qual < 1 || p->qual == 0) return 0;
	for (l = p->len - 1; l >= BWA_MIN_RDLEN - 1; --l) {
		s += trim_qual - (p->qual[l] - 33);
		if (s < 0) break;
		if (s > max) {
			max = s; max_l = l;
		}
	}
	p->clip_len = p->len = max_l + 1;
	return p->full_len - p->len;
}

static bwa_seq_t *bwa_read_bam(bwa_seqio_t *bs, int n_needed, int *n, int is_comp, int trim_qual)
{
	bwa_seq_t *seqs, *p;
	int n_seqs, l, i;
	long n_trimmed = 0, n_tot = 0;
	bam1_t *b;

	b = bam_init1();
	n_seqs = 0;
	seqs = (bwa_seq_t*)calloc(n_needed, sizeof(bwa_seq_t));
	while (bam_read1(bs->fp, b) >= 0) {
		uint8_t *s, *q;
		int go = 0;
		if ((bs->which & 1) && (b->core.flag & BAM_FREAD1)) go = 1;
		if ((bs->which & 2) && (b->core.flag & BAM_FREAD2)) go = 1;
		if ((bs->which & 4) && !(b->core.flag& BAM_FREAD1) && !(b->core.flag& BAM_FREAD2))go = 1;
		if (go == 0) continue;
		l = b->core.l_qseq;
		p = &seqs[n_seqs++];
		p->tid = -1; // no assigned to a thread
		p->qual = 0;
		p->full_len = p->clip_len = p->len = l;
		n_tot += p->full_len;
		s = bam1_seq(b); q = bam1_qual(b);
		p->seq = (ubyte_t*)calloc(p->len + 1, 1);
		p->qual = (ubyte_t*)calloc(p->len + 1, 1);
		for (i = 0; i != p->full_len; ++i) {
			p->seq[i] = bam_nt16_nt4_table[(int)bam1_seqi(s, i)];
			p->qual[i] = q[i] + 33 < 126? q[i] + 33 : 126;
		}
		if (bam1_strand(b)) { // then reverse 
			seq_reverse(p->len, p->seq, 1);
			seq_reverse(p->len, p->qual, 0);
		}
		if (trim_qual >= 1) n_trimmed += bwa_trim_read(trim_qual, p);
		p->rseq = (ubyte_t*)calloc(p->full_len, 1);
		memcpy(p->rseq, p->seq, p->len);
		seq_reverse(p->len, p->seq, 0); // *IMPORTANT*: will be reversed back in bwa_refine_gapped()
		seq_reverse(p->len, p->rseq, is_comp);
		p->name = strdup((const char*)bam1_qname(b));
		if (n_seqs == n_needed) break;
	}
	*n = n_seqs;
	if (n_seqs && trim_qual >= 1)
		fprintf(stderr, "[bwa_read_seq] %.1f%% bases are trimmed.\n", 100.0f * n_trimmed/n_tot);
	if (n_seqs == 0) {
		free(seqs);
		bam_destroy1(b);
		return 0;
	}
	bam_destroy1(b);
	return seqs;
}

#define BARCODE_LOW_QUAL 13
bwa_seq_t *bwa_read_seq(bwa_seqio_t *bs, int n_needed, int *n, int mode, int trim_qual)
{
	int rank,size;
	bwa_seq_t *seqs, *p = 0;
	kseq_t *seq = bs->ks;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int n_seqs, l, i, is_comp = mode&BWA_MODE_COMPREAD, is_64 = mode&BWA_MODE_IL13, l_bc = mode>>24;
	long n_trimmed = 0, n_tot = 0;
	// initialize to zero
	*n = 0;

	if (l_bc > 15) {
		fprintf(stderr, "[%s] the maximum barcode length is 15.\n", __func__);
		return 0;
	}
	if (bs->is_bam) return bwa_read_bam(bs, n_needed, n, is_comp, trim_qual); // l_bc has no effect for BAM input

	n_seqs = 0;
	seqs = (bwa_seq_t*)calloc(n_needed, sizeof(bwa_seq_t));
	while ((l = kseq_read(seq)) >= 0) {
		if (getPos(bs) > bs->endPos)
			break;
		if (is_64 && p->qual)
			for (i = 0; i < seq->qual.l; ++i) seq->qual.s[i] -= 31;
		p = &seqs[n_seqs++];
		if (l_bc && seq->seq.l > l_bc) { // then trim barcode
			for (i = 0; i < l_bc; ++i)
				p->bc[i] = (seq->qual.l && seq->qual.s[i]-33 < BARCODE_LOW_QUAL)? tolower(seq->seq.s[i]) : toupper(seq->seq.s[i]);
			p->bc[i] = 0;
			for (; i < seq->seq.l; ++i)
				seq->seq.s[i - l_bc] = seq->seq.s[i];
			seq->seq.l -= l_bc; seq->seq.s[seq->seq.l] = 0;
			if (seq->qual.l) {
				for (i = l_bc; i < seq->qual.l; ++i)
					seq->qual.s[i - l_bc] = seq->qual.s[i];
				seq->qual.l -= l_bc; seq->qual.s[seq->qual.l] = 0;
			}
			l = seq->seq.l;
		} else p->bc[0] = 0;
		p->tid = -1; // no assigned to a thread
		p->qual = 0;
		p->full_len = p->clip_len = p->len = l;
		n_tot += p->full_len;
		p->seq = (ubyte_t*)calloc(p->len, 1);
		for (i = 0; i != p->full_len; ++i)
			p->seq[i] = nst_nt4_table[(int)seq->seq.s[i]];
		if (seq->qual.l) { // copy quality
			p->qual = (ubyte_t*)strdup((char*)seq->qual.s);
			if (trim_qual >= 1) n_trimmed += bwa_trim_read(trim_qual, p);
		}
		p->name = strdup((const char*)seq->name.s);
		p->rseq = (ubyte_t*)calloc(p->full_len, 1);
		memcpy(p->rseq, p->seq, p->len);
		seq_reverse(p->len, p->seq, 0); // *IMPORTANT*: will be reversed back in bwa_refine_gapped()
		seq_reverse(p->len, p->rseq, is_comp);
		trimReadName(p->name);
		//if we've read all the sequences we need to process, then get out of here!
		if (n_seqs == n_needed) break;
	}

	*n = n_seqs;

	if (n_seqs && trim_qual >= 1)
		fprintf(stderr, "Proc %d: [bwa_read_seq] %.1f%% bases are trimmed.\n", rank, 100.0f * n_trimmed/n_tot);
	if (n_seqs == 0) {
		free(seqs);
		return 0;
	}
	return seqs;
}

void bwa_free_read_seq(int n_seqs, bwa_seq_t *seqs)
{
	int i, j;
	for (i = 0; i != n_seqs; ++i) {
		bwa_seq_t *p = seqs + i;
		for (j = 0; j < p->n_multi; ++j)
			if (p->multi[j].cigar) free(p->multi[j].cigar);
		free(p->name);
		free(p->seq); free(p->rseq); free(p->qual); free(p->aln); free(p->md); free(p->multi);
		free(p->cigar);
	}
	free(seqs);
}
