function x_dot = CLOSED_LOOP_SYS(x, u, t)
global eAmp; 
global Nw; 

x_dot = zeros(1,3+Nw);

x1 = x(1);
x2 = x(2);

e = eAmp;  

g = cos(2*x1) + 2;

f = -(x1+x2)/2 + x2*(g^2)/2;

x_dot(1) = -x1 + x2;
x_dot(2) = f + g*(u+e);

x_dot(3) =  x1^2 + x2^2 + u^2;

x_dot(4) =  2*x1*e;
x_dot(5) =  2*x1*x1*e;
x_dot(6) =  2*x1*(x1^2)*e;
x_dot(7) =  2*x1*(x1^3)*e;
x_dot(8) =  2*x1*(x1^4)*e;
x_dot(9) =  2*x1*(x1^5)*e;
x_dot(10) =  2*x1*(x1^6)*e;

x_dot(11) =  2*x2*e;
x_dot(12) =  2*x2*x1*e;
x_dot(13) =  2*x2*(x1^2)*e;
x_dot(14) =  2*x2*(x1^3)*e;
x_dot(15) =  2*x2*(x1^4)*e;
x_dot(16) =  2*x2*(x1^5)*e;
x_dot(17) =  2*x2*(x1^6)*e;
