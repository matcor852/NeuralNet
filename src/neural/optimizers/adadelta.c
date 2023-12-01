#include "neural/optimizer.h"

optimizer *adadelta_init(const size_t nb_layer, layer *head, __attribute__((unused)) const double l_rate) {
	optimizer *optz = calloc(1, sizeof(optimizer));
	optz->funcs = get_optz(O_ADADELTA);
	optz->nb = nb_layer - 1;
	optz->curr_layer = optz->nb;
	optz->Mwt = malloc(sizeof(dvec *) * optz->nb);
	optz->Vwt = malloc(sizeof(dvec *) * optz->nb);
	optz->Mbt = malloc(sizeof(dvec *) * optz->nb);
	optz->Vbt = malloc(sizeof(dvec *) * optz->nb);
	layer *l = head->next;
	for(size_t i = 0; i < optz->nb; ++i) {
		optz->Mwt[i] = dvec_init(l->conns, true);
		optz->Vwt[i] = dvec_init(l->conns, true);
		optz->Mbt[i] = dvec_init(l->neurons, true);
		optz->Vbt[i] = dvec_init(l->neurons, true);
		l = l->next;
	}
	return optz;
}

double adadelta_wgd(optimizer *restrict o, const double gradient) {
	*(o->vwt) = b1 * *(o->vwt) + c1 * pow(gradient, 2);
	const double upd = sqrt(*(o->mwt) + EPSILON) / sqrt(*(o->vwt) + EPSILON) * gradient;
	*(o->mwt) = b1 * *(o->mwt) + c1 * pow(upd, 2);
	o->vwt++;
	o->mwt++;
	return upd;
}

double adadelta_bgd(optimizer *restrict o, const double gradient) {
	*(o->vbt) = b1 * *(o->vbt) + c1 * pow(gradient, 2);
	const double upd = sqrt(*(o->mbt) + EPSILON) / sqrt(*(o->vbt) + EPSILON) * gradient;
	*(o->mbt) = b1 * *(o->mbt) + c1 * pow(upd, 2);
	o->vbt++;
	o->mbt++;
	return upd;
}

void adadelta_next_layer(optimizer *restrict o) {
	o->curr_layer--;
	o->mwt = o->Mwt[o->curr_layer]->vec;
	o->vwt = o->Vwt[o->curr_layer]->vec;
	o->mbt = o->Mbt[o->curr_layer]->vec;
	o->vbt = o->Vbt[o->curr_layer]->vec;
}

void adadelta_next_iter(optimizer *restrict o) {
	o->curr_layer = o->nb - 1;
	o->mwt = o->Mwt[o->curr_layer]->vec;
	o->vwt = o->Vwt[o->curr_layer]->vec;
	o->mbt = o->Mbt[o->curr_layer]->vec;
	o->vbt = o->Vbt[o->curr_layer]->vec;
}

void adadelta_free(optimizer *o) {
	for(size_t i = 0; i < o->nb; ++i) {
		dvec_free(o->Mwt[i]);
		dvec_free(o->Vwt[i]);
		dvec_free(o->Mbt[i]);
		dvec_free(o->Vbt[i]);
	}
	free(o->Mwt);
	free(o->Vwt);
	free(o->Mbt);
	free(o->Vbt);
	free(o);
}
