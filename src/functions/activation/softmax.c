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

void softmax(const dvec *restrict input, dvec *restrict output) {
	dvec *in_c = dvec_copy(input);
	double Maxd = in_c->vec[0];
	for(double *i = in_c->vec; i < in_c->vec + in_c->size; ++i) Maxd = fmax(*i, Maxd);
	for(double *i = in_c->vec; i < in_c->vec + in_c->size; ++i) *i -= Maxd;

	double s = .0, *expd = malloc(sizeof(double) * in_c->size);
	if(!expd) errx(1, "softmax: %s", NO_MEM);
	for(double *i = in_c->vec, *e = expd; i < in_c->vec + in_c->size; ++i, ++e) {
		*e = expn(*i);
		s += *e;
	}

	for(double *o = output->vec, *e = expd; o < output->vec + output->size; ++o, ++e) *o = (*e) / (s + EPSILON);

	free(expd);
	dvec_free(in_c);
}

dvec *softmax_(const dvec *restrict vec) {
	dvec *rt = dvec_init(vec->size, false);

	dvec *in_c = dvec_copy(vec);
	double Maxd = in_c->vec[0];
	for(double *i = in_c->vec; i < in_c->vec + in_c->size; ++i) Maxd = fmax(*i, Maxd);
	for(double *i = in_c->vec; i < in_c->vec + in_c->size; ++i) *i -= Maxd;

	double s = .0, *expd = malloc(sizeof(double) * in_c->size);
	if(!expd) errx(1, "softmax_: %s", NO_MEM);
	for(double *i = in_c->vec, *e = expd; i < in_c->vec + in_c->size; ++i, ++e) {
		*e = expn(*i);
		s += *e;
	}

	for(double *r = rt->vec, *e = expd; r < rt->vec + rt->size; ++e, ++r) *r = *e * (s - *e) / (pow(s, 2) + EPSILON);

	free(expd);
	dvec_free(in_c);

	return rt;
}
