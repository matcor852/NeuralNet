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

#include "neural/layer.h"

layer *layer_init(const char *name, const size_t nb_neurons, layer *prev, const double *weights, const double *bias) {
	if(!nb_neurons) errx(2, "layer_init: empty layer (0 neuron)");

	layer *l = calloc(1, sizeof(layer));
	if(!l) errx(1, "layer_init: %s", NO_MEM);

	l->a_f = get_activation(name);
	if(!l->a_f->activate && prev) errx(2, "layer_init: no activation for a processing layer");
	if(l->a_f->activate && !prev) {
		fprintf(stderr, "layer_init: activation specified for input layer will be ignored\n");
		l->a_f = get_activation("none");
	}

	l->neurons = nb_neurons;
	l->output = dvec_init(nb_neurons, false);
	l->prev = prev;
	if(prev) {
		l->conns = prev->neurons * l->neurons;
		l->input = dvec_init(nb_neurons, false);
		l->bias = bias ? dvec_from(bias, nb_neurons) : dvec_init(nb_neurons, true);
		if(!weights) {
			l->weights = dvec_init(l->conns, false);
			l->a_f->initializer(l->weights, prev->neurons, l->neurons);
		} else l->weights = dvec_from(weights, l->conns);
	}

	return l;
}

void layer_activate(layer *restrict l) {
	if(!l) return;
	if(l->prev) {
		for(double *lI = l->input->vec, *lB = l->bias->vec; lI < l->input->vec + l->input->size; ++lI, ++lB) *lI = *lB;

		for(double *pO = l->prev->output->vec, *lW = l->weights->vec; pO < l->prev->output->vec + l->prev->output->size;
			++pO)
			for(double *lI = l->input->vec; lI < l->input->vec + l->input->size; ++lI, ++lW) *lI += *pO * *lW;
	}
	l->a_f->activate(l->input, l->output);
	layer_activate(l->next);
}

void layer_free(layer *l) {
	if(!l) return;
	dvec_free(l->output);
	if(l->prev) {
		dvec_free(l->input);
		dvec_free(l->weights);
		dvec_free(l->bias);
	}
	free(l);
}
