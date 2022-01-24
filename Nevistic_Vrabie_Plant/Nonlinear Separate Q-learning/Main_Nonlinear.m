
% Main program

clc;
clear all;
close all;

format

t0 = 0; tf = 25;  %tf is set in a way that mod(tf,Tadp) = 0

ts = 0.001; % Inner Sampling Time
Ts = 0.001; % Outer Sampling Time
Tadp = 0.1; % Iteration Period

InnerIter = fix(Ts/ts); 
OuterIter = fix(Tadp/Ts); 
ADPIter = fix((tf-t0)/Tadp);

global wcNow; % Last updated critic weights
global waNow; % Last updated actor weights
global eAmp;  % Amplitude of explorations that may be used in the construction of the explorations.
global IsCriticActor;   % 0: Critic is updated now, 1: Actor is updated now.

t = t0:Ts:tf;
t_temp = 0:ts:Ts;

rk8_init;

x0 = [0.5 -0.5]';
wc = [-1 3 3/2]';
wa = [-3/2 -3/2]';

wcNow = wc;
waNow = wa;

x_temp = x0';
u_temp = CONTROLLER(x0);

x = [x0 zeros(2,OuterIter*ADPIter-1)];
u = [u_temp zeros(1,OuterIter*ADPIter-1)];

V = 0;
W = [0 0];

X = [x0(1)^2 x0(1)*x0(2) x0(2)^2]';
dX = zeros(3,1);
dZ = zeros(2,1);
Y = [];

l = 1;  nActorLearning = 1; nCriticLearning = 1; 
r = 0;  condn = [];  
critic_learning_points = [t0 ; x0 ; u_temp];
actor_learning_points = [t0 ; x0 ; u_temp];

condn = [];
eAmp = 0;

Nc = 30; % number of data for critic update
Na = 20; % number of data for actor update

CriticRankCond = 3; %full-rank condition for critic update
ActorRankCond = 2; %full-rank condition for actor update

IsCriticActor = 0; % initially critic is updated first....
N = Nc; % number of data for updating the corresponding weight vector (default: critic)
RankCond = CriticRankCond; %full-rank condition for updating the corresponding weight vector (default: critic)
Phi = [];
eAmp_Seq = [];

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
    
    if(IsCriticActor==0) % Critic is updated now...
        Y(l) = V;
        Phi(:,l) = dX(:,l);
    else % Actor is updated now...
        dZ(:,l) = [W(1) W(2)]';
        Y(l) = wc(:,nCriticLearning)'*dX(:,l) + V;
        Phi(:,l) = dZ(:,l);
    end

    V = 0;
    W = [0 0];
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    r = rank(Phi(:,1:l))

    if(r == RankCond && l>=N)
        [Kron_U Kron_S Kron_V] = svd(Phi(:,1:l)');
        condn = [condn, Kron_S(1,1)/Kron_S(r,r)];
        Kron_S(1,1)/Kron_S(r,r)
        
        [Ssrow Sscol] = size(Kron_S);
        Inv_Kron_S = inv(Kron_S(1:r,1:r));
        Inv_Kron_S = [Inv_Kron_S zeros(r,  Ssrow-r)];
        Inv_Kron_S = [Inv_Kron_S ; zeros(Sscol-r, Ssrow)];

        if(IsCriticActor==0) % Critic is updated now...
            wcNow = -(Kron_V*Inv_Kron_S*Kron_U'*Y')';
            wc(:,nCriticLearning+1) = wcNow;
            nCriticLearning = nCriticLearning + 1;
            %%% Information for ploting the critic graphs %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            critic_learning_points(:,nCriticLearning) = [t(OUTERINDEX); x(:,OUTERINDEX); u(OUTERINDEX)];
            %%% Initialization for the next actor update %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            RankCond = ActorRankCond;
            N = Na;
            IsCriticActor = 1; % the next step is to update the actor weights...
            eAmp = 3.5;
            eAmp_Seq = [eAmp_Seq eAmp];
        else % Actor is updated now....
            waNow = -(Kron_V*Inv_Kron_S*Kron_U'*Y')';
            wa(:,nActorLearning+1) = waNow;
            nActorLearning = nActorLearning + 1;
            %%% Information for ploting the actor graphs %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            actor_learning_points(:,nActorLearning) = [t(OUTERINDEX); x(:,OUTERINDEX); u(OUTERINDEX)];
            %%% Initialization for the next critic update %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            RankCond = CriticRankCond;
            N = Nc;
            IsCriticActor = 0; % the next step is to update the critic weights...
            eAmp = 0;
        end

        Y = [];       Phi = [];
        r = 0;        l = 0;    
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

n = nCriticLearning;

while (n>1)
     plot(critic_learning_points(1, n), critic_learning_points(2, n)','b.:', 'MarkerSize', 20);
     hold on
     plot(critic_learning_points(1, n), critic_learning_points(3, n)','r.:', 'MarkerSize', 20);
     hold on
     n = n-1;
end

n = nActorLearning;

while (n>1)
     plot(actor_learning_points(1, n), actor_learning_points(2, n)','b.:', 'MarkerSize', 20);
     hold on
     plot(actor_learning_points(1, n), actor_learning_points(3, n)','r.:', 'MarkerSize', 20);
     hold on
     n = n-1;
end

title('State Trajectories for Integral Q-learning II');
xlabel('Time(sec)');
ylabel('x');
legend('x_1', 'x_2');

figure
l = 6
plot(critic_learning_points(1, 1:6), wc(1,1:6)' ,'b.--', 'LineWidth',LineW, 'MarkerSize', 20);
hold on
plot(critic_learning_points(1, 1:6), wc(2,1:6)' ,'gx--', 'LineWidth',LineW,  'MarkerSize', 9);
hold on
plot(critic_learning_points(1, 1:6), wc(3,1:6)' ,'rs--', 'LineWidth',LineW,  'MarkerSize', 9);
hold on
plot(actor_learning_points(1, 1:6), wa(1,1:6)' ,'kv--', 'LineWidth',LineW, 'MarkerSize', 9);
hold on
plot(actor_learning_points(1, 1:6), wa(2,1:6)' ,'m+--', 'LineWidth',LineW,  'MarkerSize', 9);
grid on

hl = legend('$w_{1}^{(i)}$ (Critic)', '$w_{2}^{(i)}$ (Critic)', '$w_{3}^{(i)}$ (Critic)', '$v_{1}^{(i)}$ (Actor)', '$v_{2}^{(i)}$ (Actor)');
hx = xlabel('Time ($\tau$ [s])');
hy = ylabel('Amplitude');
ht = title('Variations of the Critic and Actor Weights for Algorithm 4');

set(ht, 'Interpreter', 'latex', 'FontName', 'Times New Roman');
set(hx, 'Interpreter', 'latex', 'FontName', 'Times New Roman');
set(hy, 'Interpreter', 'latex', 'FontName', 'Times New Roman');
set(hl, 'Interpreter', 'latex', 'FontName', 'Times New Roman');

save('Separate Q-learning');

