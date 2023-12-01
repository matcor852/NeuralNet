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

static const double isrlu_alpha = .5;

void isrlu(const dvec *restrict input, dvec *restrict output) {
	for(double *o = output->vec, *i = input->vec; o < output->vec + output->size; ++i, ++o)
		*o = *i < .0 ? *i / (sqrt(1 + isrlu_alpha * pow(*i, 2)) + EPSILON) : *i;
}

dvec *isrlu_(const dvec *restrict vec) {
	dvec *rt = dvec_init(vec->size, false);
	for(double *r = rt->vec, *i = vec->vec; r < rt->vec + rt->size; ++i, ++r)
		*r = *i < .0 ? pow(1. / sqrt(1. + isrlu_alpha * pow(*i, 2.)), 3.) : 1.;
	return rt;
}
