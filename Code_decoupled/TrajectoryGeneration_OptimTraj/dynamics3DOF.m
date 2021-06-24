function dz = dynamics3DOF(z,u,p)
%
% This function computes the dynamics of a 3dof pointmass lander.
%
% INPUTS:
%   z = [7, n] = [x;y;z;dx;dy;dz;m] = state of the system
%   u = [3, n] = [Tx;Ty;Tz]
%   p = parameter struct
%       .mu = gravitational parameter
%       .g0 = gravity on earth
%       .g = gravitaty acceleration on moon
%       .Isp = specific impulse of engine
%       .R = planetary/lunar radius
% OUTPUTS:
%   dz = dz/dt = time derivative of state
%

J = p.r^2 * z(7,:);


dx = z(4,:);
dy = z(5,:);
dphi = z(6,:);
ddx   = u(1,:)./z(7,:);
ddy   = u(2,:)./z(7,:) - p.g;
ddphi = u(3,:)./J;
dm = -(vecnorm(u)) / (p.Isp*p.g0);

dz = [dx;dy;dphi;ddx;ddy;ddphi;dm];

end