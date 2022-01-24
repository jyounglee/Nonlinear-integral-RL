function x_dot = CLOSED_LOOP_SYS(x, u, t)
global eAmp;

x1 = x(1);
x2 = x(2);

e = eAmp;  

f = -(x1+x2)/2 + x2*((sin(x1))^2)/2;
g = sin(x1);

x_dot(1) = -x1 + x2;
x_dot(2) = f + g*(u+e);

x_dot(3) =  x1^2 + x2^2 + u^2;
x_dot(4) =  2*x1*sin(x1)*e;
x_dot(5) =  2*x2*sin(x1)*e;