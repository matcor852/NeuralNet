

#include "functions/activation.h"

static const double selu_alpha = 1.6732632423543772848170429916717;
static const double selu_lambda = 1.0507009873554804934193349852946;

void selu(const dvec *restrict input, dvec *restrict output) {
	for(double *o = output->vec, *i = input->vec; o < output->vec + output->size; ++i, ++o)
		*o = selu_lambda * (fmax(.0, *i) + fmin(.0, selu_alpha * (expn(*i) - 1.)));
}

dvec *selu_(const dvec *restrict vec) {
	dvec *rt = dvec_init(vec->size, false);
	for(double *r = rt->vec, *i = vec->vec; r < rt->vec + rt->size; ++i, ++r)
		*r = selu_lambda * (*i < .0 ? selu_alpha * expn(*i) : 1.);
	return rt;
}
