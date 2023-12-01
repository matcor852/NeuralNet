#include "neural/optimizer.h"

optimizer *adagrad_init(const size_t nb_layer, layer *head, const double l_rate) {
	optimizer *optz = calloc(1, sizeof(optimizer));
	optz->funcs = get_optz(O_ADAGRAD);
	optz->l_rate = l_rate;
	optz->nb = nb_layer - 1;
	optz->curr_layer = optz->nb;
	optz->Mwt = malloc(sizeof(dvec *) * optz->nb);
	optz->Mbt = malloc(sizeof(dvec *) * optz->nb);
	layer *l = head->next;
	for(size_t i = 0; i < optz->nb; ++i) {
		optz->Mwt[i] = dvec_init(l->conns, true);
		optz->Mbt[i] = dvec_init(l->neurons, true);
		l = l->next;
	}
	return optz;
}

double adagrad_wgd(optimizer *restrict o, const double gradient) {
	const double upd = o->l_rate / sqrt(*(o->mwt) + EPSILON) * gradient;
	*(o->mwt) += pow(gradient, 2);
	o->mwt++;
	return upd;
}

double adagrad_bgd(optimizer *restrict o, const double gradient) {
	const double upd = o->l_rate / sqrt(*(o->mbt) + EPSILON) * gradient;
	*(o->mbt) += pow(gradient, 2);
	o->mbt++;
	return upd;
}

void adagrad_next_layer(optimizer *restrict o) {
	o->curr_layer--;
	o->mwt = o->Mwt[o->curr_layer]->vec;
	o->mbt = o->Mbt[o->curr_layer]->vec;
}

void adagrad_next_iter(optimizer *restrict o) {
	o->curr_layer = o->nb - 1;
	o->mwt = o->Mwt[o->curr_layer]->vec;
	o->mbt = o->Mbt[o->curr_layer]->vec;
}

void adagrad_free(optimizer *o) {
	for(size_t i = 0; i < o->nb; ++i) {
		dvec_free(o->Mwt[i]);
		dvec_free(o->Mbt[i]);
	}
	free(o->Mwt);
	free(o->Mbt);
	free(o);
}
