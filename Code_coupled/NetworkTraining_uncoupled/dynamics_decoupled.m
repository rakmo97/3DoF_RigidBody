function [xdot] = dynamics_coupled(t,x,u1,u2,u3)

    % Parameters
    g = 9.81 /6.0;
    g0 = 9.81;
    Isp = 300;
    r = 1;

    J = ((r^2) * x(7));

    
%     Fx = u1(t);
%     Fy = u2(t);
    Tx = u1;
    Ty = u2;
    M  = u3;
   
    dx    = x(4);
    dy    = x(5);
    dphi  = x(6);
    ddx   = ((1/x(7)) * Tx);
    ddy   = ((1/x(7)) * Ty - g);
    ddphi = ((1/J) * M);
    
    phi = x(3);
    D = [cos(phi),-sin(phi);...
                sin(phi),cos(phi);...
                1,0] ;
    FxFy = inv(D'*D)*D' * [Tx;Ty;M];
    Fx = FxFy(1); Fy = FxFy(2);
    dm    = -sqrt(Fx^2 + Fy^2) / (Isp*g0);

    xdot = [dx,dy,dphi,ddx,ddy,ddphi,dm]';

end

