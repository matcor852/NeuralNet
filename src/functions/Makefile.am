
lib_LIBRARIES = libfunctions.a
libfunctions_a_SOURCES = activation.h activation/activation.c \
						 cost.h cost/cost.c \
						 activation/arcsinh.c \
						 activation/arctan.c \
						 activation/argmax.c \
						 activation/bentidentity.c \
						 activation/bentidentityalt.c \
						 activation/elu.c \
						 activation/erf.c \
						 activation/exponential.c \
						 activation/gaussian.c \
						 activation/gelu.c \
						 activation/hardsigmoid.c \
						 activation/hardswish.c \
						 activation/identity.c \
						 activation/invsqrt.c \
						 activation/isrlu.c \
						 activation/isru.c \
						 activation/leakyrelu.c \
						 activation/logit.c \
						 activation/loglog.c \
						 activation/logsigmoid.c \
						 activation/mish.c \
						 activation/relu.c \
						 activation/selu.c \
						 activation/serf.c \
						 activation/sigmoid.c \
						 activation/silu.c \
						 activation/sinc.c \
						 activation/sinusoid.c \
						 activation/softmax.c \
						 activation/softplus.c \
						 activation/softsign.c \
						 activation/sqnl.c \
						 activation/square.c \
						 activation/step.c \
						 activation/symmetricsigmoid.c \
						 activation/tanh.c \
						 activation/tanhshrink.c \
						 activation/triangular.c \
						 cost/logcosh.c \
						 cost/quantile.c \
						 cost/hinge.c \
						 cost/rmsle.c \
						 cost/exponential.c \
						 cost/kullbackleibler.c \
						 cost/crossentropy.c \
						 cost/hellinger.c \
						 cost/itakurasaito.c \
						 cost/mae.c \
						 cost/mbe.c \
						 cost/mse.c \
						 cost/r2.c \
						 cost/rae.c \
						 cost/rmse.c \
						 cost/rse.c

libfunctions_a_CPPFLAGS = -I$(top_srcdir)/src
libfunctions_a_CFLAGS = -std=c99 -Werror -Wall -Wextra -Wvla

libfunctions_a_LIBADD = -lm

noinst_LIBRARIES = libfunctions.a

