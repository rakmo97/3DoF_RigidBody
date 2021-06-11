function [xdot] = dynamics_openloop(t,x,u1,u2,u3)

    % Parameters
    g = 9.81 /6.0;
    g0 = 9.81;
    Isp = 300;
    r = 1;

    J = ((r^2) * x(7));

    
    Fx = u1(t);
    Fy = u2(t);
    M  = u3(t);
%     Fx = u1;
%     Fy = u2;
%     M  = u3;
    
    dx    = x(4);
    dy    = x(5);
    dphi  = x(6);
    ddx   = (1/x(7))*Fx;
    ddy   = (1/x(7))*Fy - g;
    ddphi = (1/J)*M;
    dm    = -sqrt(Fx^2 + Fy^2 + M^2) / (Isp*g0);

    xdot = [dx,dy,dphi,ddx,ddy,ddphi,dm]';

end

