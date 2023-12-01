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

#pragma once

#include "functions/activation.h"

#include <err.h>
#include <stdlib.h>
#include <time.h>

typedef struct layer {
	size_t neurons, conns;
	struct layer *prev, *next;
	dvec *bias, *weights, *input, *output;
	const struct act *a_f;
} layer;

layer *layer_init(const char *name, const size_t nb_neurons, layer *prev, const double *weights, const double *bias);

void layer_activate(layer *l);
void layer_free(layer *l);
