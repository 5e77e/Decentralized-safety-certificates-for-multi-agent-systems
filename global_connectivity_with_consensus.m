clc;
clear;
digits(5);

R = 4.5;
sigma = R^4/log(2);
gamma = -1;
epsilon = 0.1;

% x0 = [1 4 1 5]';
x0 = [1 2
     3 2
     7 2
     9 2];
% x0 = [1 2;
%     3 2;
%     5 2;
%     7 2;
%     9 2;
%     2 3;
%     2 1;
%     8 3;
%     8 1];

xmax = max(x0(:, 1));
xmin = min(x0(:, 1));
ymax = max(x0(:, 2));
ymin = min(x0(:, 2));

dim = size(x0);
n = dim(1);
m = dim(2);

x0 = reshape(x0, n*m, 1);

syms x_sym [n*m, 1] real
syms dAx [n n] real
syms dAy [n n] real
A_sym = adj_sym(x_sym, n, m);
for i = 1:n
    dAx(i, :) = jacobian(A_sym(i, :), x_sym(i))';
    dAy(i, :) = jacobian(A_sym(i, :), x_sym(i+n))';
end

ode_fcn = @(t, y) [control(y, dAx, dAy, x_sym, epsilon, n, m)];

[t, y] = ode45(ode_fcn, [0 10], x0);

% Plot the results
figure;
subplot(2, 1, 1);
plot(t, y(:, 1:n), 'LineWidth', 1.5);
xlabel('Time');
ylabel('x');

subplot(2, 1, 2);
plot(t, y(:, n+1:n*m), 'LineWidth', 1.5);
xlabel('Time');
ylabel('y');

n_steps = size(t);
n_steps = n_steps(1);
lambdat = zeros(n_steps, 1);
for i = 1:n_steps
    A = adj(y(i, :)', n, m);
    L = lapl(A);
    [l2, ~] = eig2(L);
    lambdat(i) = l2;
end

% plot lambda2(t)
figure;
plot(t, lambdat, 'LineWidth', 1.5);
xlabel('Time');
ylabel('Position');

figure;
colors = [[1 0 0]; [0 1 0]; [0 0 1]; [0 1 1]; [1 0 1]; [1 1 0]; [0 0 0]; [0 0.4470 0.7410];	 [0.4940 0.1840 0.5560]];
hold on
for i = 1:n
    plot(y(:, i), y(:, i+n), 'Color', colors(i, :), 'LineWidth', 1.5);
    scatter(x0(i), x0(i+n), 30, colors(i, :), 'filled');
end
xlabel('x');
ylabel('y');
xlim([floor(xmin-1) ceil(xmax+1)]);
ylim([floor(ymin-1) ceil(ymax+1)]);
hold off
grid on
% axis equal

%%% FUNCTIONS %%%

function u = control(x, dAx, dAy, x_sym, epsilon, n, m)
    dAx = double(subs(dAx, x_sym, x));
    dAy = double(subs(dAy, x_sym, x));
    [l2, T, beta] = dlambda(x, dAx, dAy, n, m);
    H = eye(n*m);
    f = T*x;
    A = -beta;
    b = l2 - epsilon;
    opts = optimoptions('quadprog', 'Display', 'none');
    u = quadprog(H, f, A, b, [], [], [], [], [], opts);
end

function [l2, T, beta] = dlambda(x, dAx, dAy, n, m)
    %%% PARAMETERS %%%
    gamma = -1;
    A = adj(x, n, m);

    L = lapl(A);
    T = [[L gamma*L]; [L L]];
    [l2, v2] = eig2(L);
    V = repmat(v2, 1, n);
    V = (V' - V).^2;
    

    beta = [dot(dAx', V) dot(dAy', V)];
end

function [l2, v2] = eig2(L)
    [V, D] = eig(L);
    vals = diag(D);
    ord = sort(vals);
    l2 = ord(2);
    idx = vals == l2;
    v2 = V(:, idx);
    v2 = v2(:, 1);
end


function L = lapl(A)
    D = diag(sum(A, 2));
    L = D - A;
end

function a = weight(d)
    %%% PARAMETERS %%%
    R = 4.5;
    sigma = R^4/log(2);
    if d <= R
        a = exp((R^2 - d^2)^2/sigma) - 1;
    else
        a = 0;
    end
end

function A = adj(x, n, m)
    x = reshape(x, n, m);
    A = zeros(n);
    for i = 1:n
        for j = 1:n
            if j == i
                continue
            end
            A(i, j) = weight(norm(x(i, :) - x(j, :)));
            % A(i, j) = norm(x(i, :) - x(j, :));
        end
    end
end

function y = weight_sym(d)
    %%% PARAMETERS %%%
    R = 4.5;
    sigma = R^4/log(2);
    y = piecewise(d <= R, exp(((R^2 - d^2)^2)/sigma) - 1, 0);
    y = vpa(y, 4);
end

function A = adj_sym(x, n, m)
    x = reshape(x, n, m);
    syms A [n, n] real
    for i = 1:n
        for j = 1:n
            if j == i
                A(i, j) = 0;
            else
                A(i, j) = weight_sym(norm(x(i, :) - x(j, :)));
            end
        end
    end
end