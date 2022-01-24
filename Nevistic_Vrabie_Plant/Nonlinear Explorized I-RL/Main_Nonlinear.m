%
% Main program
%
% written by Jaeyoung Lee, 2014
%

clc;
clear all;
close all;

format

t0 = 0; tf = 25;  %tf is set in a way that mod(tf,Tadp) = 0

ts = 0.001;  % Inner Sampling Time
Ts = 0.001;  % Outer Sampling Time
Tadp = 0.1;  % Iteration Period

InnerIter = fix(Ts/ts); 
OuterIter = fix(Tadp/Ts); 
ADPIter = fix((tf-t0)/Tadp);

global wcNow;
global eAmp;

t = t0:Ts:tf;
t_temp = 0:ts:Ts;

rk8_init;

x0 = [0.5 -0.5]';
wc = [-1 3 3/2]';
wcNow = wc;

x_temp = x0';
u_temp = CONTROLLER(x0);

x = [x0 zeros(2,OuterIter*ADPIter-1)];
u = [u_temp zeros(1,OuterIter*ADPIter-1)];

V = 0;
W = [0 0 0];

X = [x0(1)^2 x0(1)*x0(2) x0(2)^2]';
dX = zeros(3,1);
Y = [];

l = 1;  nLearning = 1;  r = 0;  condn = [];  
learning_points = [t0 ; x0 ; u_temp];

condn = [];
eAmp = 2.5;

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
    dX(:,l) = dX(:,l) - [W(1) W(2) W(3)]'; 
    
    Y(l) = V;
    V = 0;
    W = [0 0 0];
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    r = rank(dX(:,1:l))
    
    if(l == 15)
        eAmp = -eAmp;
    end
    
    if(r == 3 && l>=30)
        [Kron_U Kron_S Kron_V] = svd(dX(:,1:l)');
        condn = [condn, Kron_S(1,1)/Kron_S(r,r)];
        Kron_S(1,1)/Kron_S(r,r)
        
        [Ssrow Sscol] = size(Kron_S);
        Inv_Kron_S = inv(Kron_S(1:r,1:r));
        Inv_Kron_S = [Inv_Kron_S zeros(r,  Ssrow-r)];
        Inv_Kron_S = [Inv_Kron_S ; zeros(Sscol-r, Ssrow)];
        wcNow = -Kron_V*Inv_Kron_S*Kron_U'*Y';
        wc(:,nLearning+1) = wcNow;
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

title('State Trajectories for Explorized I-RL');
xlabel('Time(sec)');
ylabel('x');
legend('x_1', 'x_2');

figure
l = length(learning_points(1, :));
plot(learning_points(1, 1:nLearning), wc(1,1:nLearning)' ,'b.--', 'LineWidth',LineW, 'MarkerSize', 20);
hold on
plot(learning_points(1, 1:nLearning), wc(2,1:nLearning)' ,'gx--', 'LineWidth',LineW,  'MarkerSize', 9);
hold on
plot(learning_points(1, 1:nLearning), wc(3,1:nLearning)' ,'rs--', 'LineWidth',LineW,  'MarkerSize', 9);

legend('w_{c,1}', 'w_{c,2}', 'w_{c,3}');
xlabel('Time(sec)');
ylabel('Amplitude');
title('Evolutions of the Critic Weight w_c for Explorized I-RL');

save('Explorized I-RL');