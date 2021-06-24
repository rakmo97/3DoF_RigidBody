function f = objective(~,~,u)

Fx = u(1,:);
Fy = u(2,:);

f = Fx.^2 + Fy.^2;

end

