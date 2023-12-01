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

double LogCosh(const dvec *restrict predicted, const dvec *restrict expected) {
	double loss = .0;
	for(double *p = predicted->vec, *e = expected->vec; p < predicted->vec + predicted->size; ++p, ++e)
		loss += log(cosh(*p - *e));
	return loss / (double)predicted->size;
}

dvec *LogCosh_(const dvec *restrict predicted, const dvec *restrict expected) {
	dvec *rt = dvec_init(predicted->size, false);
	for(double *p = predicted->vec, *e = expected->vec, *r = rt->vec; p < predicted->vec + predicted->size;
		++p, ++e, ++r)
		*r = tanh(*p - *e) / (double)predicted->size;
	return rt;
}
