
SUBDIRS = tools functions neural

bin_PROGRAMS = NeuralNet
NeuralNet_SOURCES = main.c

NeuralNet_CPPFLAGS = -I$(top_srcdir)/src
NeuralNet_CFLAGS = -std=c99 -Werror -Wall -Wextra -Wvla

NeuralNet_LDADD = \
	$(top_builddir)/src/neural/libneural.a \
	$(top_builddir)/src/functions/libfunctions.a \
	$(top_builddir)/src/tools/libtools.a \
	-lm

