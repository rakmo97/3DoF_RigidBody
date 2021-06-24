function xdot = dynamics_2DoF(~,x,u)

g0 = 9.81;
g = g0/6;
Isp = 300;

m = x(5);
Fx = u(1);
Fy = u(2);


dx = x(3);
dy = x(4);
ddx = (1/m)*Fx;
ddy = (1/m)*Fy - g;
dm = -sqrt(Fx^2+Fy^2) / (Isp*g0);


xdot = [dx,dy,ddx,ddy,dm];

end

