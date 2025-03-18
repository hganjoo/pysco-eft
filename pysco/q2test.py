import numpy as np


i = 1
j = 1
k = 1
h = 1.
x = 1
y = 1
z = 1

pi = np.random.normal(size=(32,32,32))


onebyeight = 1./8
h4 = h**4

q2offd = -onebyeight*((pi[x,-1 + y,-1 + z] - pi[x,-1 + y,1 + z] - pi[x,1 + y,-1 + z] + pi[x,1 + y,1 + z])**2 
    - 16.*((pi[x,y,-1 + z] + pi[x,y,1 + z])*(pi[x,-1 + y,z] + pi[x,1 + y,z]) + pi[-1 + x,y,z]*(pi[x,-1 + y,z] + pi[x,y,-1 + z] + pi[x,y,1 + z] + pi[x,1 + y,z]) + (pi[x,-1 + y,z] + pi[x,y,-1 + z] + pi[x,y,1 + z] + pi[x,1 + y,z])*pi[1 + x,y,z]) 
    + (pi[-1 + x,y,-1 + z] - pi[-1 + x,y,1 + z] - pi[1 + x,y,-1 + z] + pi[1 + x,y,1 + z])**2 
    + (pi[-1 + x,-1 + y,z] - pi[-1 + x,1 + y,z] - pi[1 + x,-1 + y,z] + pi[1 + x,1 + y,z])**2)/(h4)

print(q2offd)
