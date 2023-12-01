
#include "neural/network.h"

int main(void) {

	// XOR samples
	const double input[4][2] = {
		{.0, .0},
		{.0, 1.},
		{1., .0},
		{1., 1.},
	};

	const double expected[4][2] = {
		{.0},
		{1.},
		{1.},
		{.0},
	};

	static const size_t in_size = 2, out_size = 1;

	network *net = network_init();
	network_addLayer(net, in_size, "none", NULL, NULL);
	network_addLayer(net, 2, "silu", NULL, NULL);
	network_addLayer(net, out_size, "sigmoid", NULL, NULL);

	nn_opts opts = {
		.in_size = in_size,
		.out_size = out_size,
		.nb_train = 4,
		.nb_valid = 4,
		.l1_norm = .0,
		.l2_norm = .0,
		.cost = get_cost("MAE"),
		.epoch = 300,
		.logs = "stats.txt",
		.optz = get_optz("RMSProp")->initialize(net->nb_layers, net->head, .1),
	};

	dvec **train_in = malloc(sizeof(dvec *) * opts.nb_train);
	dvec **train_out = malloc(sizeof(dvec *) * opts.nb_train);
	for(size_t i = 0; i < opts.nb_train; ++i) {
		train_in[i] = dvec_from(input[i], opts.in_size);
		train_out[i] = dvec_from(expected[i], opts.out_size);
	}

	opts.train_input = train_in;
	opts.train_output = train_out;
	opts.valid_input = train_in;
	opts.valid_output = train_out;

	network_display(net);
	network_train(net, &opts);
	free(network_evaluate(net, &opts, true));

	for(size_t i = 0; i < opts.nb_train; ++i) {
		dvec_free(opts.train_input[i]);
		dvec_free(opts.train_output[i]);
	}
	free(train_in);
	free(train_out);

	opts.optz->funcs->free(opts.optz);

	network_free(net);
}
