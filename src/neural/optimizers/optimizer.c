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

#include "neural/optimizer.h"

optimizer *nopti_init(__attribute__((unused)) const size_t nb_layer, __attribute__((unused)) layer *head,
					  const double l_rate) {
	optimizer *optz = calloc(1, sizeof(optimizer));
	optz->funcs = get_optz(O_NONE);
	optz->l_rate = l_rate;
	return optz;
}

double nopti_wgd(optimizer *restrict o, const double gradient) { return o->l_rate * gradient; }

double nopti_bgd(optimizer *restrict o, const double gradient) { return o->l_rate * gradient; }

void nopti_next_layer(__attribute__((unused)) optimizer *o) { return; }

void nopti_next_iter(__attribute__((unused)) optimizer *o) { return; }

void nopti_free(optimizer *o) { free(o); }

const struct optz *get_optz(const char *restrict name) {
	for(size_t i = 0; i < sizeof(optzs) / sizeof(optzs[0]); ++i)
		if(!strcmp(optzs[i].name, name)) return &optzs[i];
	errx(2, "get_optz: unknown optimizer '%s'", name);
}
