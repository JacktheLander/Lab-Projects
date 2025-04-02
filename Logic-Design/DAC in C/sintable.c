#include "parameters.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int wavetable[WAVETABLE_SIZE];
int phaseAcc;
int phaseReg;


void fill_wavetable(){
	double sample, phase;

  	for (int i = 0; i < WAVETABLE_SIZE; i++) {
  		phase = (2.0 * M_PI * (double) (i) / (double) WAVETABLE_SIZE);
  		sample = (sin(phase) + 1.0) * 0.5 * SIN_SCALE * WAVETABLE_RANGE;
  		wavetable[i] = ((int) sample) & WAVETABLE_OUTPUT_MASK;
//  		printf("i = %d, phase = %f, sample = %f, wavetable[%d], = %d\n",i, phase, sample, i, wavetable[i]);
	  }
}

int table_lookup(int phase) {
	return wavetable[(phase & PHASE_MASK) >> PHASE_FRAC_BITS] & WAVETABLE_OUTPUT_MASK;
}

void clearPhaseAcc() {
	phaseAcc = 0;
}

int incPhase() {
	phaseAcc = (phaseAcc + phaseReg) & PHASE_MASK;
	return phaseAcc;
}

int getPhase() {
	return phaseAcc >> PHASE_FRAC_BITS;
}

void setPhaseReg(int phaseinc) {
	phaseReg = phaseinc & PHASE_MASK;
//	printf("Phase Increment is %d\n",phaseinc);
}
