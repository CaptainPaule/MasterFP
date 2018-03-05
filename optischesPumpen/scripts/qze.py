#!/usr/bin/python
# -*- coding: utf-8 -*-

# define values
gf = 0.3
muB = 10**(-24)
B = 10**(-3)
MF = 0
EHyp = 10**(-24)

lin = gf * muB * B
print(lin)

quad = gf**2 * muB**2 * B**2 * (1)/(EHyp)
print(quad)