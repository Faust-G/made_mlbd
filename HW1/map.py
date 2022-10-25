#!/usr/bin/python3

from csv import reader
import sys

size = 0
mean = 0
var = 0
for line in reader(sys.stdin):
    try:
        price = float(line[9])
    except:
        continue
    var = size * var / (size + 1) + size * ((mean - 1) / (size + 1)) **2
    mean = (size * mean + price) / (size + 1)
    size += 1
print(size, mean, var, sep = '\t')
