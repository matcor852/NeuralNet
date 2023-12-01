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

#include "functions/cost.h"

double R2(const dvec *restrict predicted, const dvec *restrict expected) { return 1. - RSE(predicted, expected); }

dvec *R2_(const dvec *restrict predicted, const dvec *restrict expected) {
	dvec *rt = dvec_init(predicted->size, false);
	const double yb = dvec_mean(expected);
	double deno = .0;
	for(double *e = expected->vec; e < expected->vec + expected->size; ++e) deno += pow(*e - yb, 2.);

	for(double *p = predicted->vec, *e = expected->vec, *r = rt->vec; p < predicted->vec + predicted->size;
		++p, ++e, ++r)
		*r = 2. * (*e - *p) / (deno + EPSILON);
	return rt;
}
