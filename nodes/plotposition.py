#!/usr/bin/env python

import matplotlib.pyplot as plt
import csv

x = []
y = []

odom_x = []
odom_y = []

zed_x = []
zed_y = []

with open('../dataTxt/pos.txt','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=' ')
    for row in plots:
        x.append(row[0])
        y.append(row[1])

with open('../dataTxt/teker.txt','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=' ')
    for row in plots:
        odom_x.append(row[0])
        odom_y.append(row[1])

with open('../dataTxt/zed.txt','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=' ')
    for row in plots:
        zed_x.append(row[0])
        zed_y.append(row[1])

plt.plot(x,y, '-b', label='EKF')
plt.plot(odom_x, odom_y, '-r', label='Odom')
plt.plot(zed_x,zed_y, '-y', label='zed')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Interesting Graph\nCheck it out')
plt.legend()
plt.show()