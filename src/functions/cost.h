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

#include <float.h>
#include <math.h>

double R2(const dvec *predicted, const dvec *expected);
double MAE(const dvec *predicted, const dvec *expected);
double MSE(const dvec *predicted, const dvec *expected);
double MBE(const dvec *predicted, const dvec *expected);
double RAE(const dvec *predicted, const dvec *expected);
double RSE(const dvec *predicted, const dvec *expected);
double RMSE(const dvec *predicted, const dvec *expected);
double LogCosh(const dvec *predicted, const dvec *expected);
double Quantile(const dvec *predicted, const dvec *expected);
double Hinge(const dvec *predicted, const dvec *expected);
double RMSLE(const dvec *predicted, const dvec *expected);
double Exponential(const dvec *predicted, const dvec *expected);
double KullbackLeibler(const dvec *predicted, const dvec *expected);
double CrossEntropy(const dvec *predicted, const dvec *expected);
double Hellinger(const dvec *predicted, const dvec *expected);
double ItakuraSaito(const dvec *predicted, const dvec *expected);

dvec *R2_(const dvec *predicted, const dvec *expected);
dvec *MAE_(const dvec *predicted, const dvec *expected);
dvec *MSE_(const dvec *predicted, const dvec *expected);
dvec *MBE_(const dvec *predicted, const dvec *expected);
dvec *RAE_(const dvec *predicted, const dvec *expected);
dvec *RSE_(const dvec *predicted, const dvec *expected);
dvec *RMSE_(const dvec *predicted, const dvec *expected);
dvec *LogCosh_(const dvec *predicted, const dvec *expected);
dvec *Quantile_(const dvec *predicted, const dvec *expected);
dvec *Hinge_(const dvec *predicted, const dvec *expected);
dvec *RMSLE_(const dvec *predicted, const dvec *expected);
dvec *Exponential_(const dvec *predicted, const dvec *expected);
dvec *KullbackLeibler_(const dvec *predicted, const dvec *expected);
dvec *CrossEntropy_(const dvec *predicted, const dvec *expected);
dvec *Hellinger_(const dvec *predicted, const dvec *expected);
dvec *ItakuraSaito_(const dvec *predicted, const dvec *expected);

const struct cost *get_cost(const char *name);

static const struct cost {
	const char *name;
	double (*loss)(const dvec *, const dvec *);
	dvec *(*loss_derivate)(const dvec *, const dvec *);
} costs[] = {
	{"R²", R2, R2_},
	{"MAE", MAE, MAE_},
	{"MSE", MSE, MSE_},
	{"MBE", MBE, MBE_},
	{"RAE", RAE, RAE_},
	{"RSE", RSE, RSE_},
	{"RMSE", RMSE, RMSE_},
	{"LogCosh", LogCosh, LogCosh_},
	{"Quantile", Quantile, Quantile_},
	{"Hinge", Hinge, Hinge_},
	{"RMSLE", RMSLE, RMSLE_},
	{"Exponential", Exponential, Exponential_},
	{"KullbackLeibler", KullbackLeibler, KullbackLeibler_},
	{"CrossEntropy", CrossEntropy, CrossEntropy_},
	{"Hellinger", Hellinger, Hellinger_},
	{"Itakura-Saito", ItakuraSaito, ItakuraSaito_},
};
