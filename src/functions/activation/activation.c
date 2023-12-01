/*
   Copyright (C) 2023 Matthieu Correia <matcor852@gmail.com>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

	http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

*/

#include "functions/activation.h"

const struct act *get_activation(const char *restrict name) {
	for(size_t i = 0; i < (sizeof(acts) / sizeof(acts[0])); ++i)
		if(!strcmp(acts[i].name, name)) return &acts[i];
	errx(2, "get_activation: unknown activation function '%s'", name);
}

static inline void dist_init(dvec *restrict d, const double mean, const double std_dev) {
	for(double *a = d->vec; a < d->vec + d->size; ++a) *a = n_rand() * std_dev + mean;
}

void normal_init(dvec *restrict d, __attribute__((unused)) const size_t prev_n,
				 __attribute__((unused)) const size_t curr_n) {
	static const double mean = .0, std_deviation = 1.;
	dist_init(d, mean, std_deviation);
}

void he_init(dvec *restrict d, const size_t prev_n, __attribute__((unused)) const size_t curr_n) {
	const double mean = .0, std_deviation = sqrt(2. / (double)prev_n);
	dist_init(d, mean, std_deviation);
}

void xavier_init(dvec *restrict d, const size_t prev_n, const size_t curr_n) {
	const double mean = -sqrt(6. / (double)(prev_n + curr_n)), std_deviation = sqrt(24. / (double)prev_n + curr_n);
	dist_init(d, mean, std_deviation);
}

void s0(dvec *restrict d) {
	for(double *o = d->vec; o < d->vec + d->size; ++o) *o = (*o > .0) ? 1. : .0;
}

void s0_5(dvec *restrict d) {
	for(double *o = d->vec; o < d->vec + d->size; ++o) *o = (*o >= .5) ? 1. : .0;
}

void s_max(dvec *d) { argmax(d, d); }
