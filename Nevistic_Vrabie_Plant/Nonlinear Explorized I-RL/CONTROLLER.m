function u = CONTROLLER(x_in)
global wcNow;

u = - wcNow(2)*x_in(1)*sin(x_in(1))/2 - wcNow(3)*x_in(2)*sin(x_in(1));