#include "neural/optimizer.h"

optimizer *momentum_init(const size_t nb_layer, layer *head, const double l_rate) {
	optimizer *optz = calloc(1, sizeof(optimizer));
	optz->funcs = get_optz(O_MOMENTUM);
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

double momentum_wgd(optimizer *restrict o, const double gradient) {
	*(o->mwt) = b1 * *(o->mwt) + c1 * gradient;
	double rt = o->l_rate * *(o->mwt);
	o->mwt++;
	return rt;
}

double momentum_bgd(optimizer *restrict o, const double gradient) {
	*(o->mbt) = b1 * *(o->mbt) + c1 * gradient;
	double rt = o->l_rate * *(o->mbt);
	o->mbt++;
	return rt;
}

void momentum_next_layer(optimizer *restrict o) {
	o->curr_layer--;
	o->mwt = o->Mwt[o->curr_layer]->vec;
	o->mbt = o->Mbt[o->curr_layer]->vec;
}

void momentum_next_iter(optimizer *restrict o) {
	o->curr_layer = o->nb - 1;
	o->mwt = o->Mwt[o->curr_layer]->vec;
	o->mbt = o->Mbt[o->curr_layer]->vec;
}

void momentum_free(optimizer *o) {
	for(size_t i = 0; i < o->nb; ++i) {
		dvec_free(o->Mwt[i]);
		dvec_free(o->Mbt[i]);
	}
	free(o->Mwt);
	free(o->Mbt);
	free(o);
}
