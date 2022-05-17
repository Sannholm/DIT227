// PCG -----------------------------------------
// Adapted from:
// - https://www.reedbeta.com/blog/hash-functions-for-gpu-rendering/
// - https://www.pcg-random.org/
// - https://github.com/imneme/pcg-c/blob/83252d9c23df9c82ecb42210afed61a7b42402d7/include/pcg_variants.h#L1533

#define PCG_DEFAULT_MULTIPLIER_32 747796405
#define PCG_DEFAULT_INCREMENT_32  2891336453

void pcg_oneseq_32_step_r(inout uint state) {
    state = state * PCG_DEFAULT_MULTIPLIER_32
                    + PCG_DEFAULT_INCREMENT_32;
}

uint pcg_output_rxs_m_xs_32_32(uint state) {
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

uint pcg_oneseq_32_rxs_m_xs_32_random_r(inout uint state) {
    uint oldstate = state;
    pcg_oneseq_32_step_r(state);
    return pcg_output_rxs_m_xs_32_32(oldstate);
}

void pcg_oneseq_32_srandom_r(inout uint state, uint initstate) {
    state = 0;
    pcg_oneseq_32_step_r(state);
    state += initstate;
    pcg_oneseq_32_step_r(state);
}

// ---------------------------------------------

#define RngState uint

void initRng(inout RngState state, uint seed) {
    pcg_oneseq_32_srandom_r(state, seed);
}

uint randInt(inout RngState state) {
	return pcg_oneseq_32_rxs_m_xs_32_random_r(state);
}

uint randIntRange(inout RngState state, const uint bound) {
	uint threshold = -bound % bound;
	while (true) {
		uint r = randInt(state);
		if (r >= threshold)
			return r % bound;
	}
}

float rand(inout RngState state) {
	return ldexp(randInt(state), -32);
}