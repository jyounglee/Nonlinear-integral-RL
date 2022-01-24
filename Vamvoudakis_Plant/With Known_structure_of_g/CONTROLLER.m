function u = CONTROLLER(x_in)
global wNow;

u = [wNow(4) wNow(5)]*[cos(2*x_in(1))*x_in(1) ; x_in(1)] + [wNow(6) wNow(7)]*[cos(2*x_in(1))*x_in(2) ; x_in(2)];