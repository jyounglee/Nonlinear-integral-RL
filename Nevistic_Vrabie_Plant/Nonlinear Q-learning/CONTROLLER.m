function u = CONTROLLER(x_in)
global wNow;

u = wNow(4)*x_in(1)*sin(x_in(1)) + wNow(5)*x_in(2)*sin(x_in(1));