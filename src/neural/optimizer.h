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

#include "neural/layer.h"
#include "tools/tools.h"

#include <stdlib.h>

#define O_NONE	   "none"
#define O_MOMENTUM "Momentum"
#define O_RMSPROP  "RMSProp"
#define O_ADAM	   "Adam"
#define O_ADADELTA "Adadelta"
#define O_ADAGRAD  "Adagrad"
#define O_ADAMAX   "Adamax"
#define O_NADAM	   "Nadam"

static const double b1 = .9;
static const double b2 = .999;
static const double c1 = 1. - b1;
static const double c2 = 1. - b2;

typedef struct optimizer {
	size_t iter, nb, curr_layer;
	double l_rate, b1t, b2t;
	double *mwt, *vwt, *mbt, *vbt;
	dvec **Mwt, **Vwt, **Mbt, **Vbt;
	const struct optz *funcs;
} optimizer;

optimizer *nopti_init(const size_t nb_layer, layer *head, const double l_rate);
double nopti_wgd(optimizer *o, const double gradient);
double nopti_bgd(optimizer *o, const double gradient);
void nopti_next_layer(optimizer *o);
void nopti_next_iter(optimizer *o);
void nopti_free(optimizer *o);

optimizer *momentum_init(const size_t nb_layer, layer *head, const double l_rate);
double momentum_wgd(optimizer *o, const double gradient);
double momentum_bgd(optimizer *o, const double gradient);
void momentum_next_layer(optimizer *o);
void momentum_next_iter(optimizer *o);
void momentum_free(optimizer *o);

optimizer *rmsprop_init(const size_t nb_layer, layer *head, const double l_rate);
double rmsprop_wgd(optimizer *o, const double gradient);
double rmsprop_bgd(optimizer *o, const double gradient);
void rmsprop_next_layer(optimizer *o);
void rmsprop_next_iter(optimizer *o);
void rmsprop_free(optimizer *o);

optimizer *adam_init(const size_t nb_layer, layer *head, const double l_rate);
double adam_wgd(optimizer *o, const double gradient);
double adam_bgd(optimizer *o, const double gradient);
void adam_next_layer(optimizer *o);
void adam_next_iter(optimizer *o);
void adam_free(optimizer *o);

optimizer *adadelta_init(const size_t nb_layer, layer *head, const double l_rate);
double adadelta_wgd(optimizer *o, const double gradient);
double adadelta_bgd(optimizer *o, const double gradient);
void adadelta_next_layer(optimizer *o);
void adadelta_next_iter(optimizer *o);
void adadelta_free(optimizer *o);

optimizer *adagrad_init(const size_t nb_layer, layer *head, const double l_rate);
double adagrad_wgd(optimizer *o, const double gradient);
double adagrad_bgd(optimizer *o, const double gradient);
void adagrad_next_layer(optimizer *o);
void adagrad_next_iter(optimizer *o);
void adagrad_free(optimizer *o);

optimizer *adamax_init(const size_t nb_layer, layer *head, const double l_rate);
double adamax_wgd(optimizer *o, const double gradient);
double adamax_bgd(optimizer *o, const double gradient);
void adamax_next_layer(optimizer *o);
void adamax_next_iter(optimizer *o);
void adamax_free(optimizer *o);

optimizer *nadam_init(const size_t nb_layer, layer *head, const double l_rate);
double nadam_wgd(optimizer *o, const double gradient);
double nadam_bgd(optimizer *o, const double gradient);
void nadam_next_layer(optimizer *o);
void nadam_next_iter(optimizer *o);
void nadam_free(optimizer *o);

const struct optz *get_optz(const char *name);

static const struct optz {
	const char *name;
	optimizer *(*initialize)(const size_t nb_layer, layer *head, const double l_rate);
	double (*wgt)(optimizer *o, const double gradient);
	double (*bgt)(optimizer *o, const double gradient);
	void (*next_layer)(optimizer *o);
	void (*next_iter)(optimizer *o);
	void (*free)(optimizer *o);
} optzs[] = {
	{O_NONE, nopti_init, nopti_wgd, nopti_bgd, nopti_next_layer, nopti_next_iter, nopti_free},
	{O_MOMENTUM, momentum_init, momentum_wgd, momentum_bgd, momentum_next_layer, momentum_next_iter, momentum_free},
	{O_RMSPROP, rmsprop_init, rmsprop_wgd, rmsprop_bgd, rmsprop_next_layer, rmsprop_next_iter, rmsprop_free},
	{O_ADAM, adam_init, adam_wgd, adam_bgd, adam_next_layer, adam_next_iter, adam_free},
	{O_ADADELTA, adadelta_init, adadelta_wgd, adadelta_bgd, adadelta_next_layer, adadelta_next_iter, adadelta_free},
	{O_ADAGRAD, adagrad_init, adagrad_wgd, adagrad_bgd, adagrad_next_layer, adagrad_next_iter, adagrad_free},
	{O_ADAMAX, adamax_init, adamax_wgd, adamax_bgd, adamax_next_layer, adamax_next_iter, adamax_free},
	{O_NADAM, nadam_init, nadam_wgd, nadam_bgd, nadam_next_layer, nadam_next_iter, nadam_free},
};
