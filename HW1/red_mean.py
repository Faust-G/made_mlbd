#!/usr/bin/python3

import sys

size = 0
mean = 0
var = 0

for line in sys.stdin:
    s, m, v = map(float,line.strip().split())
    var = (size * var + s * v)/ (size + s) + size * s * ((mean - m) / (size + s)) **2
    mean = (size * mean + s * m) / (size + s)
    size += s
    
print(mean, var)

