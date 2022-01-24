function u = CONTROLLER(x_in)
global wNow; global Nw;

x1 = x_in(1);
x2 = x_in(2);
Pi1 = [x1 x1*x1 x1*(x1^2) x1*(x1^3) x1*(x1^4) x1*(x1^5) x1*(x1^6) ...
       x2 x2*x1 x2*(x1^2) x2*(x1^3) x2*(x1^4) x2*(x1^5) x2*(x1^6)]';
u = wNow(4:4+Nw-1)'*Pi1;