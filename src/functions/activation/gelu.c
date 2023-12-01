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

void gelu(const dvec *restrict input, dvec *restrict output) {
	for(double *o = output->vec, *i = input->vec; o < output->vec + output->size; ++i, ++o)
		*o = *i / 2. * (1. + tanh(sqrt(2. / PI) * (*i + .044715 * pow(*i, 3.))));
}

dvec *gelu_(const dvec *restrict vec) {
	dvec *rt = dvec_init(vec->size, false);
	for(double *r = rt->vec, *i = vec->vec; r < rt->vec + rt->size; ++i, ++r) {
		const double c1 = sqrt(2. / PI) * (*i + .044715 * pow(*i, 3.));
		*r = .5 * (1. + tanh(c1)) + .5 * *i * pow(sech(c1), 2) * sqrt(2. / PI) * (1. + 3. * .044715 * pow(*i, 2));
	}
	return rt;
}
