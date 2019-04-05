#!/usr/bin/env python
# coding: utf-8

# In[3]:


## Isacc Huang, Programming Homework 1, MATH128A

## This is Problem 2

import matplotlib.pyplot as plt                                                 
import numpy as np
import math 

def f(x):
    return np.sqrt(1+x)

def a(n):
    x=1
    while n > 1:
        y = n*x
        x = f(y)
        n = n-1
    return x

for n in np.arange(40):
    print(a(n+1))

## The limit of this sequence is 3 as n approaches to infinity
j = 3
n_vals = np.arange(40) ##total numbers in the array
d_vals = np.zeros((40)) ##the size of array or set
##for loop stores every single value of d 
for n in n_vals:
    b = (a(n+1))
    c = math.fabs(b-j)
    d = math.log(c)
    d_vals[n] = d ## sign each d to nth term in the array
    
x = np.linspace(1, 40, 200)
k = math.log(2)*x
y = -k +3

plt.plot(n_vals, d_vals, x, y)
## Config the graph
plt.title('$ln(|a_n -a|)$ vs n and y =3 -(ln2)n')
plt.xlabel('X')
plt.ylabel('Y')

## Show the graph
plt.show()

## \beta_n = 1/2^n 


# In[2]:



def fact(n):
    if n==1:
        return 1
    else: 
        return n*fact(n-1)
print(fact(7))


# In[4]:


import matplotlib.pyplot as plt                                                 
import numpy as np
import math 

## Create functions and set domain length
x = np.arange(0, 40)
k = math.log(2)*x
y = -k +3

## Plot functions and a point where they intersect
plt.plot(x, y)

## Config the graph
plt.title('y = 3-ln(2)n')
plt.xlabel('X')
plt.ylabel('Y')

## Show the graph
plt.show()

## My assumption of \beta_n is -ln(2)n


# In[22]:


## This is problem 1
import matplotlib.pyplot as plt                                                 
import numpy as np
import math 
from decimal import * ## 6 digits of precision
getcontext().prec = 6
format(math.pi, '.6g')
## define the polynomial function


def p(x):
    return c*x**3 + d*x**2 + e*x + f
## take derivative 
def g(x):
    return 3*c*x**2 + 2*d*x + e

## set p'(x) = 0 and solve for x st x_1 and x_2 using quadratic formula 

    if d > 0: 
        x1 = (-2*d + math.sqrt(4*d**2-12*c*e))/(6*c)
        x2 = (2*e)/(-2*d + math.sqrt(4*d**2-12*c*e))
    if d <= 0:
        x1 = (-2*d - math.sqrt(4*d**2-12*c*e))/(6*c)
        x2 = (2*e)/(-2*d - math.sqrt(4*d**2-12*c*e))
    
def q_1(x):
    return (-2*d + math.sqrt(4*d**2-12*c*e))/(6*c)

def q_2(x):
    return (-2*d - math.sqrt(4*d**2-12*c*e))/(6*c)

## trial 1: 
a = -1; b = 2; c= -1; d=2; e= -1; f =1
print('Trail One Result:')
if d > 0: 
    x1 = (-2*d + math.sqrt(4*d**2-12*c*e))/(6*c)
    x2 = (2*e)/(-2*d + math.sqrt(4*d**2-12*c*e))
    if x1 > x2:
        print('x_min is ', "%.6f" %x2, '. ','x_max is ', "%.6f" %x1)
    else: 
        print('x_min is ', "%.6f" %x1, '. ','x_max is ',"%.6f" % x2)
    if p(x1) > p(x2):
        print('p(x_min) = ', "%.6f" %p(x2), 'p(x_max) = ', "%.6f" %p(x1))
    else:
        print('p(x_min) = ', "%.6f" %p(x1), 'p(x_max) = ', "%.6f" %p(x2))
if d <= 0:
    x1 = (-2*d - math.sqrt(4*d**2-12*c*e))/(6*c)
    x2 = (2*e)/(-2*d - math.sqrt(4*d**2-12*c*e))
    if x1 > x2:
        print('x_min is ', "%.6f" %x2, '. ','x_max is ', "%.6f" %x1)
    else: 
        print('x_min is ', "%.6f" %x1, '. ','x_max is ', "%.6f" %x2)
    if p(x1) > p(x2):
        print('p(x_min) = ', "%.6f" %p(x2), 'p(x_max) = ', "%.6f" %p(x1))
    else:
        print('p(x_min) = ', "%.6f" %p(x1), 'p(x_max) = ', "%.6f" %p(x2))

## trial 2: 
a = 1; b = 2; c= 1; d = -2; e= -1; f =1
print('Trail Two Result:')
if d > 0: 
    x1 = (-2*d + math.sqrt(4*d**2-12*c*e))/(6*c)
    x2 = (2*e)/(-2*d + math.sqrt(4*d**2-12*c*e))
    if x1 > x2:
        print('x_min is ', "%.6f" %x2, '. ','x_max is ', "%.6f" %x1)
    else: 
        print('x_min is ', "%.6f" %x1, '. ','x_max is ', "%.6f" %x2)
    if p(x1) > p(x2):
        print('p(x_min) = ', "%.6f" %p(x2), 'p(x_max) = ', "%.6f" %p(x1))
    else:
        print('p(x_min) = ', "%.6f" %p(x1), 'p(x_max) = ', "%.6f" %p(x2))

if d <= 0:
    x1 = (-2*d - math.sqrt(4*d**2-12*c*e))/(6*c)
    x2 = (2*e)/(-2*d - math.sqrt(4*d**2-12*c*e))
    if x1 > x2:
        print('x_min is ', "%.6f" %x2, '. ','x_max is ', "%.6f" %x1)
    else: 
        print('x_min is ', "%.6f" %x1, '. ','x_max is ', "%.6f" %x2)
    if p(x1) > p(x2):
        print('p(x_min) = ', "%.6f" %p(x2), 'p(x_max) = ', "%.6f" %p(x1))
    else:
        print('p(x_min) = ', "%.6f" %p(x1), 'p(x_max) = ', "%.6f" %p(x2))

## trial 3: 
a = -2; b = 1; c= 4; d = 8; e= -4; f = -2

print('Trail Three Result:')
if d > 0: 
    x1 = (-2*d + math.sqrt(4*d**2-12*c*e))/(6*c)
    x2 = (2*e)/(-2*d + math.sqrt(4*d**2-12*c*e))
    if x1 > x2:
        print('x_min is ', "%.6f" %x2, '. ','x_max is ', "%.6f" %x1)
    else: 
        print('x_min is ', "%.6f" %x1, '. ','x_max is ', "%.6f" %x2)
    if p(x1) > p(x2):
        print('p(x_min) = ', "%.6f" %p(x2), 'p(x_max) = ', "%.6f" %p(x1))
    else:
        print('p(x_min) = ', "%.6f" %p(x1), 'p(x_max) = ', "%.6f" %p(x2))

if d <= 0:
    x1 = (-2*d - math.sqrt(4*d**2-12*c*e))/(6*c)
    x2 = (2*e)/(-2*d - math.sqrt(4*d**2-12*c*e))
    if x1 > x2:
        print('x_min is ', "%.6f" %x2, '. ','x_max is ', "%.6f" %x1)
    else: 
        print('x_min is ', "%.6f" %x1, '. ','x_max is ', "%.6f" %x2)
    if p(x1) > p(x2):
        print('p(x_min) = ', "%.6f" %p(x2), 'p(x_max) = ', "%.6f" %p(x1))
    else:
        print('p(x_min) = ', "%.6f" %p(x1), 'p(x_max) = ', "%.6f" %p(x2))
## Plot functions and a point where they intersect

ax = plt.gca()
x = np.linspace(-2,1, 256, endpoint = True)
y = 4*x**3 + 8*x**2 -4*x -2
#plt.plot(x, y)
x2 = -1.548584
y1 = -2.450447
x1 = 0.215250
y2 = 8.524521
plt.plot(x,y,x1,y1,'-o',x2,y2,'-o')
## Config the graph
plt.title('$p(x) = 4x^3+8x^2-4x-2$')
plt.xlabel('X')
plt.ylabel('p(x)')
axes = plt.gca()
axes.set_xlim([x.min(), x.max()])

## Show the graph
plt.show()

## trial 4: 
a = -1; b = 2; c = 1; d = 0; e= 1; f =-3
print('Trail Four Result:')
if d > 0: 
    x1 = (-2*d + math.sqrt(4*d**2-12*c*e))/(6*c)
    x2 = (2*e)/(-2*d + math.sqrt(4*d**2-12*c*e))
    if x1 > x2:
        print('x_min is ', "%.6f" %x2, '. ','x_max is ', "%.6f" %x1)
    else: 
        print('x_min is ', "%.6f" %x1, '. ','x_max is ', "%.6f" %x2)
    if p(x1) > p(x2):
        print('p(x_min) = ', "%.6f" %p(x2), 'p(x_max) = ', "%.6f" %p(x1))
    else:
        print('p(x_min) = ', "%.6f" %p(x1), 'p(x_max) = ', "%.6f" %p(x2))

if d <= 0 and c>0 and e >0:
    print('x_min and x_max do not exit')
    print('The local min value of function is ', "%.6f" %p(a))
    print('The local max value of function is ', "%.6f" %p(b))
else:
    x1 = (-2*d - math.sqrt(4*d**2-12*c*e))/(6*c)
    x2 = (2*e)/(-2*d - math.sqrt(4*d**2-12*c*e))
    if x1 > x2:
        print('x_min is ', "%.6f" %x2, '. ','x_max is ', "%.6f" %x1)
    else: 
        print('x_min is ', "%.6f" %x1, '. ','x_max is ', "%.6f" %x2)
    if p(x1) > p(x2):
        print('p(x_min) = ', "%.6f" %p(x2), 'p(x_max) = ', "%.6f" %p(x1))
    else:
        print('p(x_min) = ', "%.6f" %p(x1), 'p(x_max) = ', "%.6f" %p(x2))

## trial 5: 
a = -0.3; b = 0.6; c = 1.0e-14; d = 9; e= -3; f = 0
print('Trail Five Result:')
if d > 0: 
    x1 = (-2*d + math.sqrt(4*d**2-12*c*e))/(6*c)
    x2 = (2*e)/(-2*d + math.sqrt(4*d**2-12*c*e))
    if x1 > x2:
        print('x_min is ', "%.6f" %x2, '. ','x_max is ', "%.6f" %x1)
    else: 
        print('x_min is ', "%.6f" %x1, '. ','x_max is ', "%.6f" %x2)
    if p(x1) > p(x2):
        print('p(x_min) = ', "%.6f" %p(x2), 'p(x_max) = ', "%.6f" %p(x1))
    else:
        print('p(x_min) = ', "%.6f" %p(x1), 'p(x_max) = ', "%.6f" %p(x2))

if d <= 0:
    x1 = (-2*d - math.sqrt(4*d**2-12*c*e))/(6*c)
    x2 = (2*e)/(-2*d - math.sqrt(4*d**2-12*c*e))
    if x1 > x2:
        print('x_min is ', "%.6f" %x2, '. ','x_max is ', "%.6f" %x1)
    else: 
        print('x_min is ', "%.6f" %x1, '. ','x_max is ', "%.6f" %x2)
    if p(x1) > p(x2):
        print('p(x_min) = ', "%.6f" %p(x2), 'p(x_max) = ', "%.6f" %p(x1))
    else:
        print('p(x_min) = ', "%.6f" %p(x1), 'p(x_max) = ', "%.6f" %p(x2))

## trial 6: 
a = -1; b = 2; c = 0; d = 0; e= 0; f = 1.7
print('Trail Six Result:')
if d > 0: 
    x1 = (-2*d + math.sqrt(4*d**2-12*c*e))/(6*c)
    x2 = (2*e)/(-2*d + math.sqrt(4*d**2-12*c*e))
    if x1 > x2:
        print('x_min is ', "%.6f" %x2, '. ','x_max is ', "%.6f" %x1)
    else: 
        print('x_min is ', "%.6f" %x1, '. ','x_max is ', "%.6f" %x2)
    if p(x1) > p(x2):
        print('p(x_min) = ', "%.6f" %p(x2), 'p(x_max) = ', "%.6f" %p(x1))
    else:
        print('p(x_min) = ', "%.6f" %p(x1), 'p(x_max) = ', "%.6f" %p(x2))

if d <= 0 and c == 0 and e == 0:
    print('This function is a constant function.')
    print('So, the local max value and local min equals to ',f) 
else:
    x1 = (-2*d - math.sqrt(4*d**2-12*c*e))/(6*c)
    x2 = (2*e)/(-2*d - math.sqrt(4*d**2-12*c*e))
    if x1 > x2:
        print('x_min is ', "%.6f" %x2, '. ','x_max is ', "%.6f" %x1)
    else: 
        print('x_min is ', "%.6f" %x1, '. ','x_max is ', "%.6f" %x2)
    if p(x1) > p(x2):
        print('p(x_min) = ', "%.6f" %p(x2), 'p(x_max) = ', "%.6f" %p(x1))
    else:
        print('p(x_min) = ', "%.6f" %p(x1), 'p(x_max) = ', "%.6f" %p(x2))

## trial 7: 
a = 0; b = 3; c = -3; d = 9; e= -1.0e-14; f = 0
print('Trail Seven Result:')
if d > 0: 
    x1 = (-2*d + math.sqrt(4*d**2-12*c*e))/(6*c)
    x2 = (2*e)/(-2*d + math.sqrt(4*d**2-12*c*e))
    if x1 > x2:
        print('x_min is ', "%.6f" %x2, '. ','x_max is ', "%.6f" %x1)
    else: 
        print('x_min is ', "%.6f" %x1, '. ','x_max is ', "%.6f" %x2)
    if p(x1) > p(x2):
        print('p(x_min) = ', "%.6f" %p(x2), 'p(x_max) = ', "%.6f" %p(x1))
    else:
        print('p(x_min) = ', "%.6f" %p(x1), 'p(x_max) = ', "%.6f" %p(x2))

if d <= 0:
    x1 = (-2*d - math.sqrt(4*d**2-12*c*e))/(6*c)
    x2 = (2*e)/(-2*d - math.sqrt(4*d**2-12*c*e))
    if x1 > x2:
        print('x_min is ', "%.6f" %x2, '. ','x_max is ', "%.6f" %x1)
    else: 
        print('x_min is ', "%.6f" %x1, '. ','x_max is ', "%.6f" %x2)
    if p(x1) > p(x2):
        print('p(x_min) = ', "%.6f" %p(x2), 'p(x_max) = ', "%.6f" %p(x1))
    else:
        print('p(x_min) = ', "%.6f" %p(x1), 'p(x_max) = ', "%.6f" %p(x2))

## trial 8: 
a = 0; b = 1; c = 0; d = -2; e= 3; f = -1
print('Trail Eight Result:')
if d > 0: 
    x1 = (-2*d + math.sqrt(4*d**2-12*c*e))/(6*c)
    x2 = (2*e)/(-2*d + math.sqrt(4*d**2-12*c*e))
    if x1 > x2:
        print('x_min is ', "%.6f" %x2, '. ','x_max is ', "%.6f" %x1)
    else: 
        print('x_min is ', "%.6f" %x1, '. ','x_max is ', "%.6f" %x2)
    if p(x1) > p(x2):
        print('p(x_min) = ', "%.6f" %p(x2), 'p(x_max) = ', "%.6f" %p(x1))
    else:
        print('p(x_min) = ', "%.6f" %p(x1), 'p(x_max) = ', "%.6f" %p(x2))

if c<=0 and d != 0 and c ==0:
    x1 = -e/(2*d)
    print('The funciton only has x_max ', "%.6f" %x1)
    print('p(x_max) = ', "%.6f" % p(x1))

## Plot functions and a point where they intersect

ax = plt.gca()
x = np.linspace(0,1, 256, endpoint = True)
y = 0*x**3 -2*x**2 +3*x -1
x1 = 0.750000
y1 = 0.125000
plt.plot(x, y, x1,y1,'-o' )

## Config the graph
plt.title('$p(x) = -2x^2 + 3x-1$')
plt.xlabel('X')
plt.ylabel('p(x)')
axes = plt.gca()
axes.set_xlim([x.min(), x.max()])
##Show the graph
plt.show()

#Follow Dr. Jon Wilkening's instruction, he suggests run each trail one by one since lack of programming background


# In[13]:


# tells me that why we need to multiply conjugates to the quadratic formula
k =7 
k+=1e16
k-=1e16
print(k)

