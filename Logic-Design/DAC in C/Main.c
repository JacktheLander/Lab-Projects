#include <stdio.h>
#include <stdlib.h>
#include "parameters.h"
#include "sintable.h"
#include <math.h>

int outputWave[SAMPLES_PER_CYCLE];
int outputWave1[SAMPLES_PER_CYCLE];
int outputWave2[SAMPLES_PER_CYCLE];
FILE *outfile;

int phase;
double ideal;

/* run this program using the console pauser or add your own getch, system("pause") or input loop */

int main(int argc, char *argv[]) {
	printf("starting\n");
	fill_wavetable();
	clearPhaseAcc();
	setPhaseReg(PHASE_INC);
//	printf("%d = PHASE_INC\n", PHASE_INC);
//	printf("%d = PHASE_FRAC_BITS\n", PHASE_FRAC_BITS);
//	printf("%d = PHASE_FRAC_MASK\n", PHASE_FRAC_MASK);
	outfile = fopen("outputwave.csv","w");
	
/* generate 1 cycle worth of data and print it out */
	phase = getPhase();
	int s1,s2;
	int frac;
	for (int i = 0; i < SAMPLES_PER_CYCLE; i++) {
		s1 = table_lookup(phase);
		s2 = table_lookup((phase + PHASE_UNIT));
		frac = phase & PHASE_FRAC_MASK;
//		printf("%d, %d, %d, %d\n", s1, s2, frac, PHASE_UNIT);
		outputWave1[i] = (((frac * s2) + ((PHASE_UNIT-frac) * s1)) >> (WAVETABLE_OUTPUT_BITS + PHASE_FRAC_BITS - INTERP_OUTPUT_BITS)) & INTERP_OUTPUT_MASK;

		frac = (phase + 1) & PHASE_FRAC_MASK;
		outputWave2[i] = (((frac * s2) + ((PHASE_UNIT-frac) * s1)) >> (WAVETABLE_OUTPUT_BITS + PHASE_FRAC_BITS - INTERP_OUTPUT_BITS)) & INTERP_OUTPUT_MASK;
		outputWave[i] = (outputWave1[i] + outputWave2[i]) / 2;

		fprintf(outfile, "%d, %d\n", phase >> (PHASE_FRAC_BITS), outputWave[i]);
		phase = incPhase();
	}
	if (phase < PHASE_INC) phase += (WAVETABLE_SIZE<< PHASE_FRAC_BITS);
	double freqErr = (double) phase / (double) WAVETABLE_SIZE / (double) (1 << PHASE_FRAC_BITS) - 1.0;
//	printf ("Phase = %f, Range = %f\n",(double) phase, (double) WAVETABLE_SIZE);
	printf ("Frequency Error = %.3f %%\n", 100.0*freqErr);
//	do an error calculation
	double error = 0;
	for (int i = 0; i < SAMPLES_PER_CYCLE; i++) {
		ideal = sin((2.0 * M_PI * (double) (i) / (double) WAVETABLE_SIZE)  + 1.0) * 0.5 * SIN_SCALE * WAVETABLE_RANGE;
		error += ((double) outputWave[i] - ideal) * ((double) outputWave[i] - ideal) ;
	}
	printf ("SNR = %f\n", WAVETABLE_RANGE/ (error/ (double) SAMPLES_PER_CYCLE));
	
	fclose(outfile);
	return 0;
}