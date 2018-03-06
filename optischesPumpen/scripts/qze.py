#!/usr/bin/python
# -*- coding: utf-8 -*-

# define values
EHyp87 = 4.53 * 10**(-24)
EHyp85 = 2.01 * 10**(-24)

gf87 = 0.331
gf85 = 0.488

muB = 9.274 * 10**(-24)
B = 200 * 10**(-6)
MF = 0

# calc linear term for Rb85 and Rb87
lin85 = gf85 * muB * B
lin87 = gf87 * muB * B
print("linear term for Rb 85")
print(lin85)
print("linear term for Rb 87")
print(lin87)

# calc quadratic term for RB 85 and Rb87
quad85 = gf85**2 * muB**2 * B**2 * (1)/(EHyp85)
quad87 = gf87**2 * muB**2 * B**2 * (1)/(EHyp87)
print("quadratic term for Rb 85")
print(quad85)
print("quadratic term for Rb 87")
print(quad87)

# evaluate fraction of linear and quadratic term
print("lin/quad for Rb 85")
print(lin85/quad85)

print("lin/quad for Rb 87")
print(lin87/quad87)