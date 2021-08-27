function [xdot] = dynamics_coupled(t,x,u1,u2)

    % Parameters
    g = 9.81 /6.0;
    g0 = 9.81;
    Isp = 300;
    r = 1;

    J = ((r^2) * x(7));

    
%     Fx = u1(t);
%     Fy = u2(t);
    Fx = u1;
    Fy = u2;
    
    dx    = x(4);
    dy    = x(5);
    dphi  = x(6);
    ddx   = ((1/x(7)) * (Fx*cos(x(3)) - Fy*sin(x(3))));
    ddy   = ((1/x(7)) * (Fx*sin(x(3)) + Fy*cos(x(3)))) - g;
    ddphi = ((1/J) * (r*Fx));
    dm    = -sqrt(Fx^2 + Fy^2) / (Isp*g0);

    xdot = [dx,dy,dphi,ddx,ddy,ddphi,dm]';

end

