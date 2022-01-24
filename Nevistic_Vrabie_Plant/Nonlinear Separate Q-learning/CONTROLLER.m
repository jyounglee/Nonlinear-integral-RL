function u = CONTROLLER(x_in)
global waNow;

u = waNow(1)*x_in(1)*sin(x_in(1)) + waNow(2)*x_in(2)*sin(x_in(1));