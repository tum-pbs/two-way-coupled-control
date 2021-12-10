clc
close all
s = zpk('s');

% Translation
% m = 11.78;
% dt =                                      0.1;

% Rotation
m = 120;                                                        % Translation Controller
% m = 2180; % Rotation Controller
dt = 0.1;

A = [0 1 ; 0 0];
B = [0 ; 1/m];
C = [1 0 ];
D = [0 ];
G_ss = ss(A,B,C,D);
G_ss.OutputName = ['Pos'];
[num, den] = ss2tf(G_ss.A, G_ss.B, G_ss.C, G_ss.D);
G = tf(num(1,:), den); % Transfer function of position
%
%
% f = figure(1)
% f.Position = [1500 0 800 800];
% hold on
% wb = [0.3, 1, 5];
% for i=1:3
%     Gd(i) = tf([ 0 wb(i)], [1 0])  ;
%     % Gd = makeweight(10, 1, 0.01);
%     [K(i),CL,gamma,info] = loopsyn(G,Gd(i), 0.5);
%
% end
% plotoptions = bodeoptions;
% plotoptions.PhaseVisible='off';
% bodeplot(G*K(1), 'k', plotoptions);
% bodeplot(G*K(2), 'b', plotoptions);
% bodeplot(G*K(3), 'r', plotoptions);
% bodeplot(Gd(1), 'k--', plotoptions);
% bodeplot(Gd(2), 'b--', plotoptions);
% bodeplot(Gd(3), 'r--', plotoptions);
% grid on
% legend('K*G wb=0.3', 'K*G wb=1', 'K*G wb=5', "G' wb=0.3", "G' wb=1", "G' wb=5")
%%
close all

Gd = tf([ 0 0.2], [1 0])  ;
% Gd = makeweight(10, 1, 0.01);
[K,CL,gamma,info] = loopsyn(G,Gd, 0.95);

K_tf = tf(K);
system = feedback(K * G_ss,0.9);
system_tf = tf(system);

K_z = c2d(K_tf, dt, 'tustin');
% K_z = balred(K_z,4);
coefficients = cell2mat([K_z.numerator ; K_z.denominator]);
numerator = {};
denominator = {};
for i=1:length(coefficients(1,:))
    numerator{i} = sprintf('%.16e', coefficients(1,i));
    denominator{i} = sprintf('%.16e', coefficients(2,i));
end
data = jsonencode(struct('numerator', char(numerator), 'denominator', char(denominator), 'dt', dt));
file = fopen('/home/ramos/phiflow/neural_control/controller_design/ls_coeffs.json', 'w');
fprintf(file, data);
fclose(file);

% K_rom = balred(K_tf, 2);
% system_rom = feedback( K_rom* G,1);
% figure(2)
% step(system,100)


f = figure(1);
f.Position = [1500 0 1000 1500];
% subplot(4,1,1)
% plotoptions = bodeoptions;
% plotoptions.PhaseVisible = 'off';
% plotoptions.XLim = [0.000001, 10.];
% bodeplot(W1*G, plotoptions)
% yline(0,'--');
% grid on
% legend('W1 * G')

subplot(2,1,1)
plotoptions = bodeoptions;
plotoptions.XLim = [0.000001, 20.];
bodeplot(K_z, K_tf, plotoptions)
title('Comparison between discrete and continuous controller')
yline(0,'--');
grid on
legend('Discrete', 'Continuous')

subplot(2,1,2)
plotoptions = bodeoptions;
plotoptions.PhaseVisible='off';
% plotoptions.XLim = [0.000001, 20.];
bodeplot(Gd, G*K, plotoptions)
hold on
title('Difference between targeted and achieved plant')
yline(0,'--');
grid on
legend('Target', 'Achieved')
% bodeplot(feedback(G*K,1), plotoptions)

% figure(3)
% L = G*K;
% I = eye(size(L));
% S = feedback(I,L);
% T= I-S;
% plotoptions.PhaseVisible = 'off';
% bodeplot(S,'b',W1,'b--',T,'r',W2,'r--',{0.000001,100}, plotoptions)
% legend('S','W1','T','W2')



% figure(2)
% step(system)

% hold on
% impulse(system_rom,10)
% legend('Full controller', 'ROM controller')
% KS = K*S;
% T = 1-S;
% sigma(S,'b',KS,'r',T,'g',gamma/W1,'b-.',ss(gamma/W2),'r-.',gamma/W3,'g-.',{1e-3,1e3})
% legend('S','KS','T','GAM/W1','GAM/W2','GAM/W3','Location','SouthWest')
% grid
%
% system = feedback(K * G, 1);
%
% impulse(system)
% numerator = cell2mat(numerator)
% denominator = cell2mat(denominator)
%
n_coefficients = length(numerator);
n = [];
d = [];
for i=1:n_coefficients
    n(i) = str2double(numerator{i});
    d(i) = str2double(denominator{i});
end

K_new = tf(n, d, dt);
K_z
gamma

%%
close all
clc
x = 0 ;
vel = 0;
obj = 15;
% obj = 2 * pi

nt = 2000;
buffer_error = zeros(1,n_coefficients);
buffer_effort = zeros(1,n_coefficients-1);
plot_data = zeros(3, nt);

for i = 1:nt
    obj(i) = 15 ;
%         obj(i) = 3;
%     obj(i) = 15+ sin(i*dt*5) * 5;
%     obj(i) = 10;

    error = obj(i) - x;
    buffer_error(1:end-1) = buffer_error(2:end);
    buffer_error(end) = error;
    u = 0;
    for j=1:n_coefficients
        u = u + n(j) * buffer_error(n_coefficients-j+1) ;
        if j > 1
            u = u - d(j) * buffer_effort(n_coefficients-j+1);
        end
    end
    u = u / d(1)    ;
    plot_data(2,i) = u;
%     u = sign(u) * min([abs(u), 60]);
    buffer_effort(1:end-1) = buffer_effort(2:end);
    buffer_effort(end) = u;
%     u = u + sin(i*dt*0.1) * 5;
%     u = u + 1;
    vel = vel + u / m * dt;
    x = x + vel * dt;

%     x = x + vel * dt + u/m * dt^2;
%     vel = vel + u / m * dt;
%
%     x = x + 10/m * dt^2;
%     vel = vel + 10/m*dt;

    plot_data(1,i) = x;
    plot_data(3,i) = vel;
end
error
max(plot_data(2,:))
f = figure(1);
f.Position = [1500 0 1000 1000];

subplot(3,1,1)
plot((1:nt)* dt, plot_data(1,:),'b')
hold on
grid on
ylabel('x')
plot((1:nt)* dt, obj,'b--')

subplot(3,1,2)
hold on
grid on
ylabel('u')
plot((1:nt)* dt, plot_data(2,:), 'r')

subplot(3,1,3)
hold on
grid on
ylabel('v')
plot((1:nt)* dt, plot_data(3,:), 'k')

xlabel('t')
