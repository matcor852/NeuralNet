#include "neural/optimizer.h"

optimizer *adam_init(const size_t nb_layer, layer *head, const double l_rate) {
	optimizer *optz = calloc(1, sizeof(optimizer));
	optz->funcs = get_optz(O_ADAM);
	optz->l_rate = l_rate;
	optz->nb = nb_layer - 1;
	optz->curr_layer = optz->nb;
	optz->iter = 0;
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

double adam_wgd(optimizer *restrict o, const double gradient) {
	*(o->mwt) = b1 * *(o->mwt) + c1 * gradient;
	*(o->vwt) = b2 * *(o->vwt) + c2 * pow(gradient, 2);

	const double m_rect = *(o->mwt) / (1. - o->b1t);
	const double v_rect = *(o->vwt) / (1. - o->b2t);

	double rt = m_rect * o->l_rate / (sqrt(v_rect) + EPSILON);
	o->mwt++;
	o->vwt++;
	return rt;
}

double adam_bgd(optimizer *restrict o, const double gradient) {
	*(o->mbt) = b1 * *(o->mbt) + c1 * gradient;
	*(o->vbt) = b2 * *(o->vbt) + c2 * pow(gradient, 2);

	const double m_rect = *(o->mbt) / (1. - o->b1t);
	const double v_rect = *(o->vbt) / (1. - o->b2t);

	double rt = m_rect * o->l_rate / (sqrt(v_rect) + EPSILON);
	o->mbt++;
	o->vbt++;
	return rt;
}

void adam_next_layer(optimizer *restrict o) {
	o->curr_layer--;
	o->mwt = o->Mwt[o->curr_layer]->vec;
	o->vwt = o->Vwt[o->curr_layer]->vec;
	o->mbt = o->Mbt[o->curr_layer]->vec;
	o->vbt = o->Vbt[o->curr_layer]->vec;
}

void adam_next_iter(optimizer *restrict o) {
	o->iter++;
	o->b1t = pow(b1, o->iter);
	o->b2t = pow(b2, o->iter);
	o->curr_layer = o->nb - 1;
	o->mwt = o->Mwt[o->curr_layer]->vec;
	o->vwt = o->Vwt[o->curr_layer]->vec;
	o->mbt = o->Mbt[o->curr_layer]->vec;
	o->vbt = o->Vbt[o->curr_layer]->vec;
}

void adam_free(optimizer *o) {
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
