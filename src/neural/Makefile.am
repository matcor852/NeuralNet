
lib_LIBRARIES = libneural.a
libneural_a_SOURCES = layer.c layer.h network.c network.h \
					  optimizer.h optimizers/optimizer.c \
					  optimizers/adadelta.c \
					  optimizers/adagrad.c \
					  optimizers/adam.c \
					  optimizers/adamax.c \
					  optimizers/momentum.c \
					  optimizers/nadam.c \
					  optimizers/rmsprop.c

libneural_a_CPPFLAGS = -I$(top_srcdir)/src
libneural_a_CFLAGS = -std=c99 -Werror -Wall -Wextra -Wvla

libneural_a_LIBADD = -lm

noinst_LIBRARIES = libneural.a

