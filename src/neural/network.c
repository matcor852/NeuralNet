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

#include "neural/network.h"

static bool sigdown = false;

static void sig_handler(int signal) {
	if(signal == SIGINT) sigdown = true;
}

network *network_init(void) {
	srand(time(NULL));
	setlocale(LC_NUMERIC, "");
	struct sigaction act = {0};
	act.sa_handler = &sig_handler;
	if(sigaction(SIGINT, &act, NULL)) {
		fprintf(stderr, "network_init: SIGINT redirection failed\n");
		exit(1);
	}

	network *net = calloc(1, sizeof(network));
	if(!net) errx(1, "network_init: %s", NO_MEM);
	return net;
}

void network_addLayer(network *net, const size_t nb_neurons, const char *name, const double *weights,
					  const double *bias) {
	layer *l = layer_init(name, nb_neurons, net->tail, weights, bias);
	if(!net->head) net->head = l;
	else net->tail->next = l;
	net->tail = l;
	net->nb_layers++;
}

bool network_save(network *net, const char *path) {
	if(!net || !path || !file_override(path)) return false;

	FILE *fptr;
	if(!(fptr = fopen(path, "wb"))) {
		fprintf(stderr, "network_save: %s\n", strerror(errno));
		return false;
	}

	size_t len;
	fwrite(&net->nb_layers, sizeof(size_t), 1, fptr);
	for(layer *l = net->head; l; l = l->next) {
		if((len = strlen(l->a_f->name)) == 0) {
			fclose(fptr);
			fprintf(stderr, "network_save: unexpected len of activation name\n");
			return false;
		}
		fwrite(&len, sizeof(size_t), 1, fptr);
		fwrite(l->a_f->name, sizeof(char), len, fptr);
		fwrite(&(l->neurons), sizeof(size_t), 1, fptr);
		if(l != net->head) {
			fwrite(l->bias->vec, sizeof(double), l->bias->size, fptr);
			fwrite(&(l->weights->size), sizeof(size_t), 1, fptr);
			fwrite(l->weights->vec, sizeof(double), l->weights->size, fptr);
		}
	}

	fprintf(stdout, "network_save: serialized network in '%s'\n", path);
	return true;
}

network *network_load(const char *path) {
	FILE *fptr;
	if(!(fptr = fopen(path, "rb"))) {
		fprintf(stderr, "network_load: %s: '%s'\n", strerror(errno), path);
		return NULL;
	}

	size_t nb_layer = 0;
	fread(&nb_layer, sizeof(size_t), 1, fptr);
	if(!nb_layer) {
		fclose(fptr);
		fprintf(stderr, "network_load: corrupted data\n");
		return NULL;
	}

	network *net = network_init();
	char *act_name = NULL;
	size_t len = 0, nb_neurons = 0, nb_weights = 0;
	double *bias = NULL, *weights = NULL;
	for(size_t i = 0; i < nb_layer; ++i) {
		fread(&len, sizeof(size_t), 1, fptr);
		act_name = calloc(len + 1, sizeof(char));
		fread(act_name, sizeof(char), len, fptr);
		fread(&nb_neurons, sizeof(size_t), 1, fptr);

		if(i != 0) {
			bias = malloc(sizeof(double) * nb_neurons);
			fread(bias, sizeof(double), nb_neurons, fptr);
			fread(&nb_weights, sizeof(size_t), 1, fptr);
			weights = malloc(sizeof(double) * nb_weights);
			fread(weights, sizeof(double), nb_weights, fptr);
		}

		network_addLayer(net, nb_neurons, act_name, weights, bias);
		free(act_name);
		free(bias);
		free(weights);
	}

	fclose(fptr);
	return net;
}

static void network_forward(network *restrict net, const double *input, const size_t size) {
	if(net->nb_layers < 2) {
		fprintf(stderr, "network_forward: incomplete network\n");
		exit(2);
	}

	if(size != net->head->output->size) {
		fprintf(stderr, "network_forward: incompatible input size\n");
		exit(2);
	}

	dvec_feed(net->head->output, input, size);
	layer_activate(net->head->next);
}

dvec *network_predict(network *restrict net, const double *restrict input, const size_t size, const bool onehot) {
	network_forward(net, input, size);
	dvec *tmp = dvec_copy(net->tail->output);
	if(onehot) (net->tail->neurons > 1 ? s_max : net->tail->a_f->threshold)(tmp);
	return tmp;
}

metrics *network_evaluate(network *restrict net, nn_opts *restrict opts, bool display) {
	bool confusion_matrix = display;
	if(opts->out_size < 2 && display) {
		fprintf(stderr, "network_evaluate: single output network, disabling confusion matrix\n");
		confusion_matrix = false;
	}

	metrics *m = calloc(1, sizeof(metrics));
	if(!m) {
		fprintf(stderr, "network_evaluate: %s\n", NO_MEM);
		return NULL;
	}

	size_t **matrix = NULL;
	if(confusion_matrix) {
		matrix = malloc(sizeof(size_t *) * opts->out_size);
		if(!matrix) {
			fprintf(stderr, "network_evaluate: %s\n", NO_MEM);
			confusion_matrix = false;
		} else {
			for(size_t i = 0; i < opts->out_size; ++i) matrix[i] = calloc(opts->out_size, sizeof(size_t));
		}
	}

	size_t buffer_act = 0, buffer_exp = 0;
	bool acted = false, exped = false;
	size_t tp = 0, fp = 0, fn = 0, tn = 0;
	for(size_t i = 0; i < opts->nb_valid; ++i) {
		if(confusion_matrix) {
			buffer_act = buffer_exp = 0;
			acted = exped = false;
		}
		dvec *result = network_predict(net, opts->valid_input[i]->vec, opts->in_size, true);
		for(size_t j = 0; j < opts->out_size; ++j) {
			double expected = opts->valid_output[i]->vec[j];
			if(!d_equal(result->vec[j], expected))
				if(d_equal(expected, .0)) ++fp;
				else ++fn;

			else if(d_equal(expected, .0)) ++tn;
			else ++tp;
			if(confusion_matrix) {
				if(d_equal(expected, 1.)) exped = true;
				if(!exped) buffer_exp++;
				if(d_equal(result->vec[j], 1.)) acted = true;
				if(!acted) buffer_act++;
			}
		}
		if(confusion_matrix) matrix[buffer_act][buffer_exp]++;
		dvec_free(result);
	}

	m->tp = tp;
	m->tn = tn;
	m->fp = fp;
	m->fn = fn;
	m->accuracy = (double)(tp + tn) / (double)((tp + fp + tn + fn));
	m->precision = (double)tp / (double)(tp + fp);
	m->recall = (double)tp / (double)(tp + fn);
	m->specificity = (double)tn / (double)(tn + fp);
	m->NPV = (double)tn / (double)(tn + fn);
	m->miss_rate = 1. - m->recall;
	m->fall_out = 1. - m->specificity;
	m->FDR = 1. - m->precision;
	m->FOR = 1. - m->NPV;
	m->PLR = m->recall / m->fall_out;
	m->NLR = m->miss_rate / m->specificity;
	m->F1 = 2. * m->precision * m->recall / (m->precision + m->recall);

	int pad = 25;
	if(display) {
		printf("\n Network metrics:\n\n");
		printf("\t%*s: %.2lf%%\n", pad, "Recall", m->recall * 100.);
		printf("\t%*s: %.2lf%%\n", pad, "Precision", m->precision * 100.);
		printf("\t%*s: %.2lf%%\n", pad, "Specificity", m->specificity * 100.);
		printf("\t%*s: %.2lf%%\n", pad, "Negative Predictive Rate", m->NPV * 100.);

		printf("\n\t%*s: %.2lf%%\n", pad, "Fall Out", m->fall_out * 100.);
		printf("\t%*s: %.2lf%%\n", pad, "Miss Rate", m->miss_rate * 100.);
		printf("\t%*s: %.2lf%%\n", pad, "False Omission Rate", m->FOR * 100.);
		printf("\t%*s: %.2lf%%\n", pad, "False Discovery Rate", m->FDR * 100.);

		printf("\n\t%*s: %.2lf%%\n", pad, "Positive Likelihood Ratio", m->PLR * 100.);
		printf("\t%*s: %.2lf%%\n", pad, "Negative Likelihood Ratio", m->NLR * 100.);
		printf("\t%*s: %.2lf%%\n", pad, "F1-Score", m->F1 * 100.);
		printf("\t%*s: %.2lf%%\n\n", pad, "Accuracy", m->accuracy * 100.);
	}

	if(confusion_matrix) {
		printf("\n Confusion matrix:\n\n");
		double bf = .0;
		pad = 6;
		printf("%s%s%*s ", BG_BLACK, FG_RED, pad, "");
		for(size_t i = 0; i < opts->out_size; ++i) printf("%*lu ", pad, i);
		printf("        \n");
		for(size_t i = 0; i < opts->out_size; ++i) {
			printf("%s%s%*lu %s", BG_BLACK, FG_RED, pad, i, RESET);
			bf = .0;
			for(size_t j = 0; j < opts->out_size; ++j) {
				printf("%s%s%*lu %s", BG_BLACK, i == j ? FG_BLUE : FG_WHITE, pad, matrix[j][i], RESET);
				bf += matrix[j][i];
			}
			float v = (double)matrix[i][i] / (double)bf;
			printf("%s    %s%.2f%s\n", BG_BLACK, v < .6 ? FG_RED : v < .9 ? FG_YELLOW : FG_GREEN, v, RESET);
		}
		puts("");

		for(size_t i = 0; i < opts->out_size; ++i) free(matrix[i]);
		free(matrix);
	}

	return m;
}

static double network_backprop(network *restrict net, nn_opts *restrict opts, const size_t at_expected) {
	optimizer *optz = opts->optz;
	optz->funcs->next_iter(optz);

	layer *l = net->tail;
	dvec *expected = opts->train_output[at_expected];

	double l1 = .0, l2 = .0;
	bool l1b = !d_equal(.0, opts->l1_norm), l2b = !d_equal(.0, opts->l2_norm);

	double error = opts->cost->loss(l->output, expected);
	dvec *cost_out = opts->cost->loss_derivate(l->output, expected);
	dvec *out_in = l->a_f->derivate(l->input);

	dvec *legacy = dvec_init(l->prev->neurons, true);
	bool bias_done = false;

	for(double *pO = l->prev->output->vec, *leg = legacy->vec, *w = l->weights->vec;
		pO < l->prev->output->vec + l->prev->neurons; ++pO, ++leg) {
		for(double *cO = cost_out->vec, *oI = out_in->vec, *b = l->bias->vec; cO < cost_out->vec + l->neurons;
			++cO, ++oI, ++w, ++b) {
			const double ml = *cO * *oI;
			*leg += ml * *w;
			if(l1b) l1 += fabs(*w);
			if(l2b) l2 += pow(*w, 2.);
			*w -= optz->funcs->wgt(optz, ml * *pO)
				  + optz->l_rate * ((*w >= 0 ? 1 : -1) * opts->l1_norm + 2. * opts->l2_norm * *w);
			if(!bias_done) *b -= optz->funcs->bgt(optz, ml);
		}
		bias_done = true;
	}
	dvec_free(cost_out);
	dvec_free(out_in);

	dvec *tmp_legacy = NULL;
	double *t_leg = NULL;
	for(l = l->prev; l != net->head; l = l->prev) {
		bool next_leg_needed = l->prev != net->head;
		if(next_leg_needed) {
			tmp_legacy = dvec_init(l->prev->neurons, true);
			t_leg = tmp_legacy->vec;
		}
		out_in = l->a_f->derivate(l->input);
		bias_done = false;
		optz->funcs->next_layer(optz);
		for(double *w = l->weights->vec, *pO = l->prev->output->vec; pO < l->prev->output->vec + l->prev->output->size;
			++pO) {
			for(double *leg = legacy->vec, *oI = out_in->vec, *b = l->bias->vec; leg < legacy->vec + legacy->size;
				++leg, ++oI, ++w, ++b) {
				const double ml = *leg * *oI;
				if(next_leg_needed) *t_leg += ml * *w;
				if(l1b) l1 += fabs(*w);
				if(l2b) l2 += pow(*w, 2.);
				*w -= optz->funcs->wgt(optz, ml * *pO)
					  + optz->l_rate * ((*w >= 0 ? 1 : -1) * opts->l1_norm + 2. * opts->l2_norm * *w);
				if(!bias_done) *b -= optz->funcs->bgt(optz, ml);
			}
			bias_done = true;
			if(next_leg_needed) ++t_leg;
		}
		dvec_free(legacy);
		legacy = tmp_legacy;
		dvec_free(out_in);
	}

	return error + opts->l1_norm * l1 + opts->l2_norm * l2;
}

void network_train(network *restrict net, nn_opts *restrict opts) {
	if(!opts || !opts->train_input || !opts->valid_input || !opts->train_output || !opts->valid_output || !opts->cost
	   || !opts->optz) {
		fprintf(stderr, "network_train: incomplete train parameters\n");
		exit(2);
	}
	if(!net || net->nb_layers < 2) {
		fprintf(stderr, "network_train: incomplete network\n");
		exit(2);
	}
	if(opts->in_size != net->head->neurons || opts->out_size != net->tail->neurons) {
		fprintf(stderr, "network_train: incompatible trainset\n");
		exit(2);
	}

	FILE *f = NULL;
	if(opts->logs) {
		if(!file_override(opts->logs))
			fprintf(stderr, "network_train: '%s' not writable; disabling logs\n", opts->logs);
		else f = fopen(opts->logs, "w");
	}

	int pad = 1 + log10(opts->epoch);
	puts("");
	struct timespec start, stop;
	static const double sec_div = 10E9;
	double error = DBL_MAX, tmp_time = DBL_EPSILON;
	size_t e = 0, i = 0;
	clock_gettime(CLOCK_REALTIME, &start);
	for(; e <= opts->epoch && !sigdown; ++e) {
		// shuffling train set prevent order related bias
		dvec_shuffle(opts->train_input, opts->train_output, opts->nb_train);
		for(i = 0; i < opts->nb_train && !sigdown; ++i) {
			network_forward(net, opts->train_input[i]->vec, opts->in_size);
			error = network_backprop(net, opts, i);
			if(f) fprintf(f, "%lf\n", error);
			clock_gettime(CLOCK_REALTIME, &stop);
			tmp_time = (stop.tv_sec - start.tv_sec) + (double)(stop.tv_nsec - start.tv_nsec) / sec_div;
			printf("\r[%*lu/%lu] loss: %lf\t{%'.2f it/s | %'.2f epoch/s}          ", pad, e, opts->epoch, error,
				   (double)(e * opts->nb_train + i) / tmp_time, (double)(e) / tmp_time);
		}
	}
	printf("\r[%*lu/%lu] loss: %lf\t{%'.2f it/s | %'.2f epoch/s}          ", pad, e > opts->epoch ? opts->epoch : e,
		   opts->epoch, error, (double)(e * opts->nb_train + i) / tmp_time, (double)(e) / tmp_time);

	if(f) fclose(f);
	puts("\n");
}

void network_display(network *restrict net) {
	if(!net) {
		fprintf(stderr, "network_display: uninitialized network\n");
		return;
	}

	printf("\n________________________________________________________________________\n\n");
	printf("\033[;3;4mNetwork configuration\033[0m:\n\n");
	printf("    %'lu layer%s\n\n", net->nb_layers, net->nb_layers > 1 ? "s" : "");

	size_t nbW = 0;
	size_t nbB = 0;

	struct layer *l = net->head;
	for(unsigned int i = 0; l; ++i, l = l->next) {
		if(!l->prev && l->next) printf("    \033[;1;3m‣ Input Layer:\n");
		else if(!l->next) printf("    \033[;1;3m‣ Output Layer:\n");
		else printf("    \033[;1;3m‣ Hidden Layer %'u:\n", i);
		printf("\033[0m");
		printf("        %'lu neuron%s\n", l->neurons, l->neurons > 1 ? "s" : "");
		printf("        Activation function: \033[;3m%s\033[0m\n\n", l->a_f->name);
		if(l->prev) {
			nbW += l->weights->size;
			nbB += l->neurons;
		}
	}

	const size_t nbP = nbW + nbB;
	printf("    \033[;1m%'lu\033[0m trainable parameter%s: \033[;1m%'lu\033[0m weight%s + \033[;1m%'lu\033[0m bias%s\n",
		   nbP, nbP > 1 ? "s" : "", nbW, nbW > 1 ? "s" : "", nbB, nbB > 1 ? "es" : "");
	printf("________________________________________________________________________\n\n");
}

void network_free(network *net) {
	if(!net) return;
	layer *l = net->head;
	while(l) {
		layer *tmp = l->next;
		layer_free(l);
		l = tmp;
	}
	free(net);
}
