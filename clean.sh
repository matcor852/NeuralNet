#!/bin/sh

make distclean
cat .gitignore | xargs rm -rf
find . -name Makefile.in -delete
find . -name \*.gcno -delete
find . -name \*.gcda -delete
find . -type d -name .deps -delete
git clean -dxi

