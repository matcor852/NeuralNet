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

#define _POSIX_C_SOURCE 200112L

#include "functions/cost.h"
#include "neural/layer.h"
#include "neural/optimizer.h"

#include <errno.h>
#include <locale.h>
#include <signal.h>

typedef struct network {
	size_t nb_layers;
	layer *head, *tail;
} network;

typedef struct nn_opts {
	size_t epoch, epoch_interval;
	size_t in_size, out_size;
	size_t nb_train, nb_valid;
	dvec **train_input, **train_output;
	dvec **valid_input, **valid_output;
	double l1_norm, l2_norm;
	const struct cost *cost;
	const char *logs;
	optimizer *optz;
} nn_opts;

typedef struct metrics {
	size_t tp, fp, fn, tn;
	double accuracy;
	double precision, recall, specificity;
	double miss_rate, fall_out;
	double FDR, FOR;
	double PLR, NLR;
	double NPV, F1;
} metrics;

network *network_init(void);
void network_addLayer(network *net, const size_t nb_neurons, const char *name, const double *weights,
					  const double *bias);

bool network_save(network *net, const char *path);
network *network_load(const char *path);

void network_train(network *net, nn_opts *opts);
metrics *network_evaluate(network *net, nn_opts *opts, bool display);
dvec *network_predict(network *net, const double *input, const size_t size, const bool onehot);

void network_display(network *net);
void network_free(network *net);
