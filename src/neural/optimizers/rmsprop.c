#include "neural/optimizer.h"

optimizer *rmsprop_init(const size_t nb_layer, layer *head, const double l_rate) {
	optimizer *optz = calloc(1, sizeof(optimizer));
	optz->funcs = get_optz(O_RMSPROP);
	optz->l_rate = l_rate;
	optz->nb = nb_layer - 1;
	optz->curr_layer = optz->nb;
	optz->Vwt = malloc(sizeof(dvec *) * optz->nb);
	optz->Vbt = malloc(sizeof(dvec *) * optz->nb);
	layer *l = head->next;
	for(size_t i = 0; i < optz->nb; ++i) {
		optz->Vwt[i] = dvec_init(l->conns, true);
		optz->Vbt[i] = dvec_init(l->neurons, true);
		l = l->next;
	}
	return optz;
}

double rmsprop_wgd(optimizer *restrict o, const double gradient) {
	*(o->vwt) = b1 * *(o->vwt) + c1 * pow(gradient, 2);
	double rt = o->l_rate / sqrt(*(o->vwt) + EPSILON) * gradient;
	o->vwt++;
	return rt;
}

double rmsprop_bgd(optimizer *restrict o, const double gradient) {
	*(o->vbt) = b1 * *(o->vbt) + c1 * pow(gradient, 2);
	double rt = o->l_rate / sqrt(*(o->vbt) + EPSILON) * gradient;
	o->vbt++;
	return rt;
}

void rmsprop_next_layer(optimizer *restrict o) {
	o->curr_layer--;
	o->vwt = o->Vwt[o->curr_layer]->vec;
	o->vbt = o->Vbt[o->curr_layer]->vec;
}

void rmsprop_next_iter(optimizer *restrict o) {
	o->curr_layer = o->nb - 1;
	o->vwt = o->Vwt[o->curr_layer]->vec;
	o->vbt = o->Vbt[o->curr_layer]->vec;
}

void rmsprop_free(optimizer *o) {
	for(size_t i = 0; i < o->nb; ++i) {
		dvec_free(o->Vwt[i]);
		dvec_free(o->Vbt[i]);
	}
	free(o->Vwt);
	free(o->Vbt);
	free(o);
}
