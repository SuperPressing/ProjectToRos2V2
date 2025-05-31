import math
x_t = []
y_t = []
ore_t = []
time_t = []
v_t = []
xv_t = []
yv_t = []
xv_t[i] = v_t[i]*math.cos(ore_t[i])
yv_t[i] = v_t[i]*math.sin(ore_t[i])
w_t = []
t = 0
x = 0
y = 0
ore = 0
v = 0
w = 0
xv = v*math.cos(ore)
yv = v*math.sin(ore)
e1 = x[i]-x_t
e2 = y[i]-y_t
e3 = ore[i] - ore_t
de1 = xv_t[i]-xv
de2 = yv_t[i]-yv
de3 = ore_t[i]-ore

k1 = 1
k2 = 1

u1 = -v_t[i]+(v_t[i]*(e1*math.cos(ore_t[i])+e2*math.sin(ore_t[i])))/(e2*math.cos(ore_t[i]+e3)+e2*math.sin(ore_t[i]+e3))-k1(e1*math.cos(ore_t[i]+e3)+e2*math.sin(ore_t[i]+e3))
u2 = -k2*e3


v_control = v_t + u1
w_control = v_t + u2