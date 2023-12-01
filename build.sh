#!/bin/sh

autoreconf --install
./configure "$1"
