
% Main program

clc;
clear all;
close all;

format

t0 = 0; tf = 6;  %tf is set in a way that mod(tf,Tadp) = 0

ts = 0.001; % Inner Sampling Time
Ts = 0.001; % Outer Sampling Time
Tadp = 0.025; % Good Tadp... (set this with eAmp...)

InnerIter = fix(Ts/ts); 
OuterIter = fix(Tadp/Ts); 
ADPIter = fix((tf-t0)/Tadp);

global wNow;
global eAmp;

t = t0:Ts:tf;
t_temp = 0:ts:Ts;

rk8_init;

x0 = [0 0]';
wc = [1 1 1]';
wa = [-1/2 -1 -1 -2]';

wNow = [wc' wa']';

x_temp = x0';
u_temp = CONTROLLER(x0);

x = [x0 zeros(2,OuterIter*ADPIter-1)];
u = [u_temp zeros(1,OuterIter*ADPIter-1)];

V = 0;
W = [0 0 0 0];

X = [x0(1)^2 x0(1)*x0(2) x0(2)^2]';
dX = zeros(3,1);
dZ = zeros(7,1);
Y = [];

l = 1;  nLearning = 1;  r = 0;  condn = [];  
learning_points = [t0 ; x0 ; u_temp];

condn = [];
eAmp = 1; % Good Tadp... (set this with Tadp...)

for i = 1:ADPIter
    for j = 1:OuterIter
        for k = 1:InnerIter
            t_now = (((i-1)*OuterIter + j-1)*InnerIter + k-1)*ts;
            [x_temp, V, W] = CalculateNext(x_temp, V, W, t_now, ts);
            u_temp = CONTROLLER(x_temp);
        end
        OUTERINDEX = (i-1)*OuterIter+j;
        x(:,OUTERINDEX) = x_temp;
        u(OUTERINDEX) = CONTROLLER(x_temp);
    end
    
    %%% Data Aquisition for Performing Least Squares %%%%%%%%%%%%%%%%%%%%%%
    dX(:,l) = [x_temp(1)^2 x_temp(1)*x_temp(2) x_temp(2)^2]' - X;
    X = dX(:,l) + X;
    dZ(:,l) = [dX(:,l)' W(1) W(2) W(3) W(4)]';
    Y(l) = V;
    V = 0;
    W = [0 0 0 0];
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    r = rank(dZ(:,1:l))
    
    if(l==20)
        eAmp = -eAmp;
    end
    
    if(r == 7 && l>=40)
        [Kron_U Kron_S Kron_V] = svd(dZ(:,1:l)');
        condn = [condn, Kron_S(1,1)/Kron_S(r,r)];
        Kron_S(1,1)/Kron_S(r,r)
        
        [Ssrow Sscol] = size(Kron_S);
        Inv_Kron_S = inv(Kron_S(1:r,1:r));
        Inv_Kron_S = [Inv_Kron_S zeros(r,  Ssrow-r)];
        Inv_Kron_S = [Inv_Kron_S ; zeros(Sscol-r, Ssrow)];
        wNow = -(Kron_V*Inv_Kron_S*Kron_U'*Y')';
        wc(:,nLearning+1) = wNow(1:3);
        wa(:,nLearning+1) = wNow(4:7);
        nLearning = nLearning+1;
        
        %%% Information for ploting the graphs %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        learning_points(:,nLearning) = [t(OUTERINDEX); x(:,OUTERINDEX); u(OUTERINDEX)];
        
        %%% Initialization for the next update %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        Y = [];
        r = 0;        l = 0;
        eAmp = -eAmp;    
    end
    l = l+1;
end

LineW = 1.5;
t = t(1:length(t)-1);

%% Ploting the trj's of states %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure
plot(t,x(1,:), 'b', 'LineWidth',LineW);
hold on
plot(t,x(2,:), 'r--', 'LineWidth',LineW);

n = nLearning;
while (n>1)
     plot(learning_points(1, n), learning_points(2, n)','b.:', 'MarkerSize', 20);
     hold on
     plot(learning_points(1, n), learning_points(3, n)','r.:', 'MarkerSize', 20);
     hold on
     n = n-1;
end

ht = title('State Trajectories for Integral $Q$-Learning I (Algorithm 3)');
hx = xlabel('Time ($\tau$ [s])');
hy = ylabel('$x$');
hl = legend('$x_1(\tau)$', '$x_2(\tau)$');
grid on
set(ht, 'Interpreter', 'latex', 'FontName', 'Times New Roman');
set(hx, 'Interpreter', 'latex', 'FontName', 'Times New Roman');
set(hy, 'Interpreter', 'latex', 'FontName', 'Times New Roman');
set(hl, 'Interpreter', 'latex', 'FontName', 'Times New Roman');

figure
l = length(learning_points(1, :));
plot(learning_points(1, 1:nLearning), wc(1,1:nLearning)' ,'b.--', 'LineWidth',LineW, 'MarkerSize', 20);
hold on
plot(learning_points(1, 1:nLearning), wc(2,1:nLearning)' ,'gx--', 'LineWidth',LineW,  'MarkerSize', 9);
hold on
plot(learning_points(1, 1:nLearning), wc(3,1:nLearning)' ,'rs--', 'LineWidth',LineW,  'MarkerSize', 9);
hold off

hl = legend('$w_{1}^{(i)}$ (Critic)', '$w_{2}^{(i)}$ (Critic)', '$w_{3}^{(i)}$ (Critic)');
hx = xlabel('Time ($\tau$ [s])');
hy = ylabel('Amplitude');
ht = title('Variations of the Critic Weights for Algorithm 3');

set(ht, 'Interpreter', 'latex', 'FontName', 'Times New Roman');
set(hx, 'Interpreter', 'latex', 'FontName', 'Times New Roman');
set(hy, 'Interpreter', 'latex', 'FontName', 'Times New Roman');
set(hl, 'Interpreter', 'latex', 'FontName', 'Times New Roman');

figure
l = length(learning_points(1, :));
plot(learning_points(1, 1:nLearning), wa(1,1:nLearning)' ,'kv--', 'LineWidth',LineW, 'MarkerSize',  9);
hold on
plot(learning_points(1, 1:nLearning), wa(2,1:nLearning)' ,'m+--', 'LineWidth',LineW,  'MarkerSize', 9);
grid on
plot(learning_points(1, 1:nLearning), wa(3,1:nLearning)' ,'co--', 'LineWidth',LineW,  'MarkerSize', 9);
grid on
plot(learning_points(1, 1:nLearning), wa(4,1:nLearning)' ,'yp--', 'LineWidth',LineW,  'MarkerSize', 9);
grid on

hl = legend('$v_{1}^{(i)}$ (Actor)', '$v_{2}^{(i)}$ (Actor)', '$v_{3}^{(i)}$ (Actor)', '$v_{4}^{(i)}$ (Actor)');
hx = xlabel('Time ($\tau$ [s])');
hy = ylabel('Amplitude');
ht = title('Variations of the Actor Weights for Algorithm 3');

set(ht, 'Interpreter', 'latex', 'FontName', 'Times New Roman');
set(hx, 'Interpreter', 'latex', 'FontName', 'Times New Roman');
set(hy, 'Interpreter', 'latex', 'FontName', 'Times New Roman');
set(hl, 'Interpreter', 'latex', 'FontName', 'Times New Roman');

save('Q-learning with g');