m = 180; 
I = 6540;
dt = 0.05;% G = 1 / (m * s^2);


A = [0 1 ; 0 0]; 
B = [0 ; 1/m]; 
B_rot = [0; 1/I];
C = [1 0 ]; 
D = [0 ]; 
G_ss = ss(A,B,C,D); 
G_ss.OutputName = ['Pos'];
[num, den] = ss2tf(G_ss.A, G_ss.B, G_ss.C, G_ss.D);
G = tf(num(1,:), den); % Transfer function of position 
G_discrete = c2d(G, dt, 'tustin');

G_ss_rot = ss(A,B_rot,C,D); 
G_ss_rot.OutputName = ['Ang'];
[num, den] = ss2tf(G_ss_rot.A, G_ss_rot.B, G_ss_rot.C, G_ss_rot.D);
G_rot = tf(num(1,:), den); % Transfer function of position 
G_discrete_rot = c2d(G, dt, 'tustin');

%% 
% m = I;
Kp = 10;
Ki = 0.5; 
Kd = 100;

Kp_print = sprintf('%.16e', Kp);    
Ki_print = sprintf('%.16e', Ki);    
Kd_print = sprintf('%.16e', Kd);    

% data = jsonencode(struct('Kp', char(Kp_print), 'Kd', char(Kd_print),'Ki', char(Ki_print) , 'dt', dt));
% file = fopen('/home/ramos/phiflow/neural_control/controller_coefficients.json', 'w'); 
% fprintf(file, data);
% fclose(file);


% close all
clc
x = 0 ;
vel = 0;
% obj = 30;
obj = 1;

last_error = obj-x; 
nt = 2000;
plot_data = zeros(3, nt);
int_error = 0;
for i = 1:nt
    error = obj - x;
    int_error = int_error + (error + last_error)/2. * dt;
    derror = (error - last_error) / dt;
    last_error = error;
    u = Kp * error + Ki * int_error + Kd * derror; 
    plot_data(2,i) = u;
    u = u + 10;
    acc = u / m;
    vel = vel + acc * dt; 
    x = x + vel * dt;     
    plot_data(1,i) = x; 
    plot_data(3,i) = vel; 
end

f = figure(1); 

f.Position = [1500 0 1000 1000];

subplot(3,1,1)
hold on 
ylabel('x')
plot((1:nt)* dt, plot_data(1,:), 'b--')
yline(10,'k-.')
legend('Loop shaping', 'PID') 
subplot(3,1,2)
hold on 
ylabel('u')
plot((1:nt)* dt, plot_data(2,:), 'r--')
legend('Loop shaping', 'PID') 
subplot(3,1,3)
hold on 
ylabel('v')

plot((1:nt)* dt, plot_data(3,:), 'k--')
legend('Loop shaping', 'PID') 