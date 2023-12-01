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

#include "tools/tools.h"

#include <err.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>

void sigmoid(const dvec *input, dvec *output);
void Tanh(const dvec *input, dvec *output);
void arctan(const dvec *input, dvec *output);
void softsign(const dvec *input, dvec *output);
void isru(const dvec *input, dvec *output);
void isrlu(const dvec *input, dvec *output);
void sqnl(const dvec *input, dvec *output);
void relu(const dvec *input, dvec *output);
void leakyrelu(const dvec *input, dvec *output);
void elu(const dvec *input, dvec *output);
void selu(const dvec *input, dvec *output);
void gelu(const dvec *input, dvec *output);
void silu(const dvec *input, dvec *output);
void mish(const dvec *input, dvec *output);
void serf(const dvec *input, dvec *output);
void softplus(const dvec *input, dvec *output);
void bentidentity(const dvec *input, dvec *output);
void sinusoid(const dvec *input, dvec *output);
void sinc(const dvec *input, dvec *output);
void gaussian(const dvec *input, dvec *output);
void identity(const dvec *input, dvec *output);
void step(const dvec *input, dvec *output);
void softmax(const dvec *input, dvec *output);
void argmax(const dvec *input, dvec *output);
void square(const dvec *input, dvec *output);
void exponential(const dvec *input, dvec *output);
void loglog(const dvec *input, dvec *output);
void hardswish(const dvec *input, dvec *output);
void invsqrt(const dvec *input, dvec *output);
void triangular(const dvec *input, dvec *output);
void hardsigmoid(const dvec *input, dvec *output);
void symmetricsigmoid(const dvec *input, dvec *output);
void logit(const dvec *input, dvec *output);
void logsigmoid(const dvec *input, dvec *output);
void arcsinh(const dvec *input, dvec *output);
void bentidentityalt(const dvec *input, dvec *output);
void tanhshrink(const dvec *input, dvec *output);
void erfb(const dvec *input, dvec *output);

dvec *sigmoid_(const dvec *vec);
dvec *Tanh_(const dvec *vec);
dvec *arctan_(const dvec *vec);
dvec *softsign_(const dvec *vec);
dvec *isru_(const dvec *vec);
dvec *isrlu_(const dvec *vec);
dvec *sqnl_(const dvec *vec);
dvec *relu_(const dvec *vec);
dvec *leakyrelu_(const dvec *vec);
dvec *elu_(const dvec *vec);
dvec *selu_(const dvec *vec);
dvec *gelu_(const dvec *vec);
dvec *silu_(const dvec *vec);
dvec *mish_(const dvec *vec);
dvec *serf_(const dvec *vec);
dvec *softplus_(const dvec *vec);
dvec *bentidentity_(const dvec *vec);
dvec *sinusoid_(const dvec *vec);
dvec *sinc_(const dvec *vec);
dvec *gaussian_(const dvec *vec);
dvec *identity_(const dvec *vec);
dvec *step_(const dvec *vec);
dvec *softmax_(const dvec *vec);
dvec *argmax_(const dvec *vec);
dvec *square_(const dvec *vec);
dvec *exponential_(const dvec *vec);
dvec *loglog_(const dvec *vec);
dvec *hardswish_(const dvec *vec);
dvec *invsqrt_(const dvec *vec);
dvec *triangular_(const dvec *vec);
dvec *hardsigmoid_(const dvec *vec);
dvec *symmetricsigmoid_(const dvec *vec);
dvec *logit_(const dvec *vec);
dvec *logsigmoid_(const dvec *vec);
dvec *arcsinh_(const dvec *vec);
dvec *bentidentityalt_(const dvec *vec);
dvec *tanhshrink_(const dvec *vec);
dvec *erfb_(const dvec *vec);

const struct act *get_activation(const char *name);

void normal_init(dvec *d, const size_t prev_n, const size_t curr_n);
void he_init(dvec *d, const size_t prev_n, const size_t curr_n);
void xavier_init(dvec *d, const size_t prev_n, const size_t curr_n);

void s0(dvec *d);
void s0_5(dvec *d);
void s_max(dvec *d);

static const struct act {
	const char *name;
	void (*activate)(const dvec *, dvec *);
	dvec *(*derivate)(const dvec *);
	void (*initializer)(dvec *, const size_t prev_n, const size_t curr_n);
	void (*threshold)(dvec *);
} acts[] = {
	{"none", NULL, NULL, NULL, NULL},
	{"sigmoid", sigmoid, sigmoid_, xavier_init, s0_5},
	{"tanh", Tanh, Tanh_, xavier_init, s0},
	{"arctan", arctan, arctan_, xavier_init, s0},
	{"softsign", softsign, softsign_, xavier_init, s0},
	{"ISRU", isru, isru_, xavier_init, s0},
	{"ISRLU", isrlu, isrlu_, xavier_init, s0},
	{"SQNL", sqnl, sqnl_, xavier_init, s_max},
	{"relu", relu, relu_, he_init, s_max},
	{"leakyrelu", leakyrelu, leakyrelu_, he_init, s_max},
	{"elu", elu, elu_, he_init, s_max},
	{"selu", selu, selu_, he_init, s_max},
	{"gelu", gelu, gelu_, he_init, s_max},
	{"silu", silu, silu_, he_init, s_max},
	{"mish", mish, mish_, he_init, s_max},
	{"serf", serf, serf_, he_init, s_max},
	{"soft+", softplus, softplus_, he_init, s_max},
	{"BentIdentity", bentidentity, bentidentity_, he_init, s_max},
	{"sinusoid", sinusoid, sinusoid_, normal_init, s0},
	{"sinc", sinc, sinc_, normal_init, s_max},
	{"gaussian", gaussian, gaussian_, normal_init, s_max},
	{"identity", identity, identity_, normal_init, s_max},
	{"step", step, step_, normal_init, s0_5},
	{"softmax", softmax, softmax_, normal_init, s_max},
	{"argmax", argmax, argmax_, normal_init, s_max},
	{"square", square, square_, normal_init, s_max},
	{"exponential", exponential, exponential_, normal_init, s_max},
	{"loglog", loglog, loglog_, normal_init, s_max},
	{"hardswish", hardswish, hardswish_, normal_init, s_max},
	{"invsqrt", invsqrt, invsqrt_, normal_init, s_max},
	{"triangular", triangular, triangular_, normal_init, s_max},
	{"hardsigmoid", hardsigmoid, hardsigmoid_, normal_init, s_max},
	{"symmetricsigmoid", symmetricsigmoid, symmetricsigmoid_, normal_init, s_max},
	{"logit", logit, logit_, normal_init, s_max},
	{"logsigmoid", logsigmoid, logsigmoid_, normal_init, s_max},
	{"arcsinh", arcsinh, arcsinh_, normal_init, s_max},
	{"bentidentityalt", bentidentityalt, bentidentityalt_, normal_init, s_max},
	{"tanhshrink", tanhshrink, tanhshrink_, normal_init, s_max},
	{"erf", erfb, erfb_, normal_init, s_max},
};
