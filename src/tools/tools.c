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

#include "tools/tools.h"

const double EPSILON = 10E-8;

double u_rand(void) { return ((double)(rand()) + 1.) / ((double)(RAND_MAX) + 1.); }

double n_rand(void) {
	double u1 = u_rand(), u2 = u_rand();
	return cos(2. * PI * u2) * sqrt(-2. * log(u1));
}

__attribute__((hot)) double expn(const double op) {
	double rt = exp(op);
	return isinf(rt) ? op : rt;
}

bool d_equal(double a, double b) { return fabs(a - b) <= ((fabs(a) > fabs(b) ? fabs(b) : fabs(a)) * DBL_EPSILON); }

double sech(const double op) { return 1. / cosh(op); }

bool file_override(const char *restrict path) {
	struct stat sb;
	if(stat(path, &sb) == 0) {
		bool replace = false;
		if(S_ISREG(sb.st_mode)) {
			fprintf(stdout, "file_override: '%s' already exists, replace it (y/n) ? ", path);
			fflush(stdout);
			replace = fgetc(stdin) == 'y';
		} else fprintf(stderr, "file_override: '%s' is a directory\n", path);
		if(!replace) {
			fprintf(stderr, "file_override: cancelling network serialization\n");
			return false;
		}
	}
	return true;
}

__attribute__((hot)) dvec *dvec_init(const size_t size, const bool zeros) {
	dvec *d = malloc(sizeof(dvec));
	if(!d) errx(1, "dvec_init: %s", NO_MEM);
	d->vec = zeros ? calloc(size, sizeof(double)) : malloc(sizeof(double) * size);
	if(!d->vec) errx(1, "dvec_init: %s", NO_MEM);
	d->size = size;
	return d;
}

dvec *dvec_copy(const dvec *restrict d) {
	dvec *n = dvec_init(d->size, false);
	memmove(n->vec, d->vec, d->size * sizeof(double));
	return n;
}

void dvec_feed(dvec *restrict d, const double *restrict feed, const size_t size) {
	memmove(d->vec, feed, size * sizeof(double));
}

dvec *dvec_from(const double *restrict vec, const size_t size) {
	dvec *n = dvec_init(size, false);
	memmove(n->vec, vec, size * sizeof(double));
	return n;
}

double *dvec_export(const dvec *restrict d) {
	double *rt = malloc(sizeof(double) * d->size);
	memmove(rt, d->vec, d->size * sizeof(double));
	return rt;
}

void dvec_shuffle(dvec **left, dvec **right, const size_t size) {
	const size_t n = rand() % size;
	for(dvec **l = left, **r = right; l < left + n; ++l, ++r) {
		const size_t i = rand() % size;
		dvec *tmp_l = *l, *tmp_r = *r;
		*l = left[i];
		left[i] = tmp_l;
		*r = right[i];
		right[i] = tmp_r;
	}
}

void dvec_print(const dvec *restrict d) {
	puts("");
	int pad = 1 + log10(d->size);
	for(size_t i = 0; i < d->size; ++i)
		printf("%*lu: %s%lf\033[0m\n", pad, i, d_equal(d->vec[i], .0) ? "\033[0;31m" : "", d->vec[i]);
	puts("");
}

double dvec_sum(const dvec *restrict d) {
	double sum = .0;
	for(double *m = d->vec; m < d->vec + d->size; ++m) sum += *m;
	return sum;
}

double dvec_min(const dvec *restrict d) {
	double min = d->vec[0];
	for(double *m = d->vec; m < d->vec + d->size; ++m) min = fmin(*m, min);
	return min;
}

double dvec_max(const dvec *restrict d) {
	double max = d->vec[0];
	for(double *m = d->vec; m < d->vec + d->size; ++m) max = fmax(*m, max);
	return max;
}

double dvec_mean(const dvec *restrict d) {
	double mean = .0;
	for(double *m = d->vec; m < d->vec + d->size; ++m) mean += *m;
	return mean / (double)d->size;
}

double dvec_variance(const dvec *restrict d) {
	const double mean = dvec_mean(d);
	double var = .0;
	for(double *m = d->vec; m < d->vec + d->size; ++m) var += pow(mean - *m, 2.);
	return var / (double)d->size;
}

double dvec_std_deviation(const dvec *restrict d) { return sqrt(dvec_variance(d)); }

void dvec_free(dvec *d) {
	free(d->vec);
	free(d);
}
