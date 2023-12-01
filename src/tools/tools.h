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

#include <err.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#define PI	   3.14159265358979323846L
#define E	   2.71828182845904523536L
#define NO_MEM "memory exhausted"

#define RESET	  "\033[0m"
#define BG_BLACK  "\033[40m"
#define FG_RED	  "\033[31m"
#define FG_BLUE	  "\033[34m"
#define FG_YELLOW "\033[33m"
#define FG_GREEN  "\033[32m"
#define FG_WHITE  "\033[37m"

extern const double EPSILON;

typedef struct dvec {
	double *vec;
	size_t size;
} dvec;

double u_rand(void);
double n_rand(void);
double expn(const double op);
double sech(const double op);
bool d_equal(double a, double b);

bool file_override(const char *path);

dvec *dvec_init(const size_t size, const bool zeros);
dvec *dvec_copy(const dvec *d);
void dvec_feed(dvec *d, const double *feed, const size_t size);
dvec *dvec_from(const double *vec, const size_t size);
double *dvec_export(const dvec *d);
void dvec_shuffle(dvec **left, dvec **right, const size_t size);
void dvec_print(const dvec *d);

double dvec_sum(const dvec *d);
double dvec_min(const dvec *d);
double dvec_max(const dvec *d);
double dvec_mean(const dvec *d);
double dvec_variance(const dvec *d);
double dvec_std_deviation(const dvec *d);

void dvec_free(dvec *d);
