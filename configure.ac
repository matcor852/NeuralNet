AC_INIT([NeuralNet], [1.0], [matcor852@gmail.com])
AC_CONFIG_SRCDIR([src/main.c])
AC_MSG_NOTICE([Configure $PACKAGE_NAME $PACKAGE_VERSION])
AM_INIT_AUTOMAKE([foreign subdir-objects])

AM_SILENT_RULES([yes])

AC_PROG_CC
AM_PROG_AR
AC_PROG_RANLIB

AX_COMPILER_FLAGS([], [], [], [-std=c99 -pedantic -Werror -Wall -Wextra -Wvla])

AC_CONFIG_FILES([
    Makefile
    src/Makefile
    src/tools/Makefile
    src/functions/Makefile
    src/neural/Makefile
])

AC_OUTPUT
