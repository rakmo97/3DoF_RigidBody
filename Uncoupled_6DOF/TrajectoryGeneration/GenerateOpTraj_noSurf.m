%% Omkar S. Mulekar
% Optimal Trajectory Generator for 3DOF rigid body Lunar Landings
% - Given specified upper and lower bounds for randomized initial
% conditions, this script will generate a specified number (nTrajs) of
% thrust optimal trajectories, i.e. trajectories that minimize the
% magnitude of thrust integrated over time.


clear all
close all 
clc
% %%


%% Generation settings
nTrajs = 500; % Number of trajectories to generate
plotting = 0; % Plot things or no?
saveout = ['d',datestr(now,'yyyymmdd_HHoMM'),'_genTrajs','.mat'];

% Lower and upper values for random initial conditions
%       [x,    y,     z,   dx,   dy,   dz,   phi,theta,psi,     p,q,r,             m]
lower = [750, -500,  750, -0.5, -0.5, -0.5, -(pi/8)*ones(1,3), -1e-4*ones(1,3)];
upper = [1000, 500, 1000, -0.25, 0.5,  0.5,  (pi/8)*ones(1,3),  1e-4*ones(1,3)];

% Target State [x,y,z,dx,dy,dz,phi,theta,psi,p,q,r,m]
target = [0,0,0,0,0,0,0,0,0,0,0,0];

% Preallocations
N_steps = 100;
surfFunctionOut = cell(nTrajs,1);
objectiveOut = zeros(nTrajs,2);
Jout = zeros(nTrajs,1);
stateOut = zeros(N_steps,14,nTrajs);
ctrlOut = zeros(N_steps,6,nTrajs);
runTimeOut = zeros(nTrajs,1);
stateFinal = zeros(nTrajs,13);

for i = 1:nTrajs
    run ../../OpenOCL/ocl.m 
    % Solve OCP
    whilecount = 0;
    err_count = 0;
    while whilecount == err_count
        try   
    
%             [gridPoints, objective, surfFunction] = produceRandomSurface();
%             surfFunctionOut{i} = surfFunction;
%             objectiveOut(i,:) = objective;

            %% parameters

            conf = struct;
            conf.g = 9.81/6; % m/s2
            conf.g0 = 9.81; % m/s2
            conf.Isp = 300;
            conf.a = 1;
            conf.b = 1;
            conf.c = 1.5;
%             conf.objective = objective;
%             conf.surfFunc = surfFunction;



            %% Setup Solver
            varsfun    = @(sh) landervarsfun(sh, conf);
            daefun     = @(sh,x,z,u,p) landereqfun(sh,x,z,u,conf);
            pathcosts  = @(ch,x,~,u,~) landerpathcosts(ch,x,u,conf);
%             terminalcosts  = @(ch,x,p) landerterminalcosts(ch,x,conf);
%             gridconstraints = @(ch,k,K,x,p) landergridconstraints(ch,k,K,x,conf);


%             solver = ocl.Problem([], varsfun, daefun, pathcosts,'terminalcost', terminalcosts,...
%                 'gridconstraints',gridconstraints,'N',N);

            solver = ocl.Problem([], varsfun, daefun, pathcosts, 'N',N_steps);

            %% Populate Solver Settings
            % Parameters
            solver.setParameter('g'    , conf.g    );
            solver.setParameter('g0'   , conf.g0   );
            solver.setParameter('a'    , conf.a    );
            solver.setParameter('b'    , conf.b    );
            solver.setParameter('c'    , conf.c    );

            % Generate Initial Conditions and Target [x,y,phi,dx,dy,dphi,m]
            m0 = 500;
            r0 = lower+(upper-lower).*rand(12,1)';
%             r0 = [983.9327, -42.1137, 810.1196, -0.3090, 0.2593, 0.2406, 0.1914, -0.3095, 0.1426, 0, -0.0001, -0.0001];
            
            % Set Initial Conditions
            solver.setInitialBounds( 'x'    ,   r0(1)   );
            solver.setInitialBounds( 'y'    ,   r0(2)   );
            solver.setInitialBounds( 'z'    ,   r0(3)   );
            solver.setInitialBounds( 'dx'   ,   r0(4)   );
            solver.setInitialBounds( 'dy'   ,   r0(5)   );
            solver.setInitialBounds( 'dz'   ,   r0(6)   );
            solver.setInitialBounds( 'phi'  ,   r0(7)   ); % roll
            solver.setInitialBounds( 'theta',   r0(8)   ); % pitch
            solver.setInitialBounds( 'psi'  ,   r0(9)   ); % yaw
            solver.setInitialBounds( 'p'    ,   r0(10)   ); % roll rate
            solver.setInitialBounds( 'q'    ,   r0(11)   ); % pitch rate
            solver.setInitialBounds( 'r'    ,   r0(12)   ); % yaw rate
            solver.setInitialBounds( 'm'    ,   m0      ); % mass


            % Set Target State
            solver.setEndBounds( 'x' ,    target(1) );
            solver.setEndBounds( 'y' ,    target(2) );
            solver.setEndBounds( 'z' ,    target(3) );
            solver.setEndBounds( 'dx'  ,  target(4) );
            solver.setEndBounds( 'dy'  ,  target(5) );
            solver.setEndBounds( 'dz' ,   target(6) );
            solver.setEndBounds( 'phi',   target(7) );
            solver.setEndBounds( 'theta', target(8) );
            solver.setEndBounds( 'psi',   target(9) );
            solver.setEndBounds( 'p',     target(10) );
            solver.setEndBounds( 'q',     target(11) );
            solver.setEndBounds( 'r',     target(12) );



    
    
            %% Run Solver
            disp(['Starting trajectory ',num2str(i),' of ',num2str(nTrajs),'....'])

            tic
            initialGuess    = solver.getInitialGuess();
            [solution,times] = solver.solve(initialGuess);
            timeToRun = toc;
            
            if ~strcmp(solver.solver.stats.return_status,'Solve_Succeeded')
                error('Optimal Solution Not Found, Retrying...')
            end
%             
            %% Process solutions
            % Grab Times
            ts = times.states.value;
            tc = times.controls.value;
            ratio = (length(ts)-1)/length(tc); % Ratio of ctrl times to state times

            % Pull out states
            x     = solution.states.x.value;
            y     = solution.states.y.value;
            z   = solution.states.z.value;
            dx    = solution.states.dx.value;
            dy    = solution.states.dy.value;
            dz  = solution.states.dz.value;
            phi  = solution.states.phi.value;
            theta  = solution.states.theta.value;
            psi  = solution.states.psi.value;
            p  = solution.states.p.value;
            q  = solution.states.q.value;
            r  = solution.states.r.value;
            m  = solution.states.m.value;

            xa     = solution.states.x.value;
            ya     = solution.states.y.value;
            za   = solution.states.z.value;
            dxa    = solution.states.dx.value;
            dya    = solution.states.dy.value;
            dza  = solution.states.dz.value;
            phia  = solution.states.phi.value;
            thetaa  = solution.states.theta.value;
            psia  = solution.states.psi.value;
            pa  = solution.states.p.value;
            qa  = solution.states.q.value;
            ra  = solution.states.r.value;
            ma  = solution.states.m.value;


%             Pull out controls
            Fx = solution.controls.Fx.value;
            Fy = solution.controls.Fy.value;
            Fz = solution.controls.Fz.value;
            L  = solution.controls.L.value;
            M  = solution.controls.M.value;
            N  = solution.controls.N.value;


%             % Define indexes of states that align with control values
            idxs = 1:ratio:length(ts)-1;

%             % Separate states by available controls
            x       = x(idxs);
            y       = y(idxs);
            z       = z(idxs);
            dx      = dx(idxs);
            dy      = dy(idxs);
            dz      = dz(idxs);
            phi     = phi(idxs);
            theta   = theta(idxs);
            psi     = psi(idxs);
            p       = p(idxs);
            q       = q(idxs);
            r       = r(idxs);
            m       = m(idxs);


            % Calculate Costs
            L_F = Fx.^2 + Fy.^2 + Fz.^2 + L.^2 + M.^2 + N.^2;
            J_F = trapz(tc,L_F);
        %     J_t = tc(end);

            J_path = J_F;
%             J_term = norm([x(end),y(end)]-objective);
%             J_total = J_path + J_term;

        %     disp(['Force min cost is ',num2str(J_F)])
            disp(['Path Cost is  ', num2str(J_path)])
%             disp(['Term cost is ', num2str(J_term)])
%             disp(['Total cost is ', num2str(J_total)])

            % Save off outputs
            Jout(i,:) = [J_path];
            stateOut(:,:,i) = [tc',x',y',z',dx',dy',dz',phi',theta',psi',p',q',r',m'];
            ctrlOut(:,:,i) = [Fx',Fy',Fz',L',M',N'];
            runTimeOut(i) = timeToRun;
            stateFinal(i,:) = [xa(end),ya(end),za(end),dxa(end),dya(end),dza(end),phia(end),theta(end),psia(end),pa(end),qa(end),ra(end),ma(end)];
             
            if plotting
                % Plot x,y,z trajectory
                figure(1);
                plot3(x(1),y(1),z(1),'rx','MarkerSize',10)
                hold on
                grid on
                plot3(solution.states.x.value,...
                   solution.states.y.value,...
                   solution.states.z.value,'Color','b','LineWidth',1.5);
                plot3(x(end),y(end),z(end),'bo','MarkerSize',10)
                % plot3(objective(1),objective(2),objective(3),'c+','MarkerSize',10)
                % surf(gridPoints(:,:,1),gridPoints(:,:,2),gridPoints(:,:,3))
                alpha 0.5
                xlabel('x[m]');ylabel('y[m]');zlabel('z[m]');
                legend('Starting Point','Trajectory','Ending Point','Objective','Surface','location','best')


                % Plot thrust profiles
                figure(2);
                subplot(2,3,1)
                hold on
                plot(tc,Fx,'g')
                title('Controls')
                ylabel('F_1 [N]')
                subplot(2,3,2)
                hold on
                plot(tc,Fy,'b')
                ylabel('F_2 [N]')
                subplot(2,3,3)
                hold on
                plot(tc,Fz,'r')
                ylabel('F_3 [N]')
                subplot(2,3,4)
                hold on
                plot(tc,L,'r')
                ylabel('L [Nm]')
                hold off
                xlabel('Time [s]')
                subplot(2,3,5)
                hold on
                plot(tc,M,'r')
                ylabel('M [Nm]')
                hold off
                xlabel('Time [s]')
                subplot(2,3,6)
                hold on
                plot(tc,N,'r')
                ylabel('N [Nm]')
                hold off
                xlabel('Time [s]')                

                figure(3);
                subplot(3,1,1)
                hold on
                plot(tc,x,'g')
                title('Position vs Time')
                ylabel('x [m]')
                subplot(3,1,2)
                hold on
                plot(tc,y,'b')
                ylabel('y [m]')
                subplot(3,1,3)
                hold on
                plot(tc,z,'r')
                ylabel('z [m]')
                hold off
                xlabel('Time [s]')

                figure(4);
                subplot(3,1,1)
                hold on
                plot(tc,rad2deg(phi),'g')
                title('Euler Angles vs Time')
                ylabel('\phi [deg]')
                subplot(3,1,2)
                hold on
                plot(tc,rad2deg(theta),'b')
                ylabel('\theta [deg]')
                subplot(3,1,3)
                hold on
                plot(tc,rad2deg(psi),'r')
                ylabel('\psi [deg]')
                hold off
                xlabel('Time [s]')

                figure(5)
                subplot(3,1,1)
                hold on
                plot(tc,dx,'g')
                title('Speed vs Time')
                ylabel('v_x [m/s]')
                hold on
                subplot(3,1,2)
                plot(tc,dy,'b')
                ylabel('v_y [m/s]')
                subplot(3,1,3)
                hold on
                plot(tc,dz,'r')
                ylabel('v_z [m/s]')
                hold off
                xlabel('Time [s]')

                figure(6);
                subplot(3,1,1)
                hold on
                hold on
                plot(tc,rad2deg(p),'g')
                title('Body rates vs Time')
                ylabel('p [deg/s]')
                subplot(3,1,2)
                hold on
                plot(tc,rad2deg(q),'b')
                ylabel('q [deg/s]')
                subplot(3,1,3)
                hold on
                 plot(tc,rad2deg(r),'r')
                ylabel('r [deg/s]')
                hold off
                xlabel('Time [s]')
            end
            
        catch
            disp('Optimal Solution Not Found, Retrying...');
            err_count = err_count+1;
        end
        whilecount = whilecount+1;
    end


    disp(['Finished trajectory ',num2str(i),' of ',num2str(nTrajs)])
    
    
%     clearvars -except nTrajs plotting saveout lower upper target N surfFunctionOut objectiveOut Jout stateOut ctrlOut runTimeOut stateFinal
%     clear solver conf;
    
end
%%
fprintf("\n\nTrajectory Generation Complete!\nSaving Variables to .mat file...\n")
disp(['Filename: ',saveout])
save(saveout,'surfFunctionOut','objectiveOut','Jout','stateOut','ctrlOut','runTimeOut','stateFinal');
fprintf("\nProgram Complete!\n")
disp(['at ',datestr(now,'yyyymmdd_HHoMMSS')])

%% Solver Functions
function landervarsfun(sh, c)


    % Define States
    sh.addState('x');
    sh.addState('y');
    sh.addState('z','lb',0);
    sh.addState('dx');
    sh.addState('dy');
    sh.addState('dz');
    sh.addState('phi'  ,'lb',-pi/4,'ub',pi/4);
    sh.addState('theta','lb',-pi/4,'ub',pi/4);
    sh.addState('psi');
    sh.addState('p');
    sh.addState('q');
    sh.addState('r');
    sh.addState('m');

    Fmax = 15000;
    % Define Controls
    sh.addControl('Fx', 'lb', -Fmax, 'ub', Fmax);  % Force [N]
    sh.addControl('Fy', 'lb', -Fmax, 'ub', Fmax);  % Force [N]
    sh.addControl('Fz', 'lb',  0   , 'ub', Fmax);  % Force [N]
    sh.addControl('L' , 'lb', -Fmax, 'ub', Fmax);  % Force [N]
    sh.addControl('M' , 'lb', -Fmax, 'ub', Fmax);  % Force [N]
    sh.addControl('N' , 'lb', -Fmax, 'ub', Fmax);  % Force [N]


    % System Parameters
    sh.addParameter('g');
    sh.addParameter('g0');
    sh.addParameter('r');
    sh.addParameter('Isp')
    sh.addParameter('objective');
    sh.addParameter('surfFunc');
    sh.addParameter('a');
    sh.addParameter('b');
    sh.addParameter('c');

end

function landereqfun(sh,x,~,u,c) % https://charlestytler.com/quadcopter-equations-motion/


    Ix = (x.m/5) * (c.b^2 + c.c^2);
    Iy = (x.m/5) * (c.a^2 + c.c^2);
    Iz = (x.m/5) * (c.a^2 + c.b^2);
    

    
    % Equations of Motion
    sh.setODE( 'x'     , x.dx);
    sh.setODE( 'y'     , x.dy);
    sh.setODE( 'z'     , x.dz);
    sh.setODE( 'dx'    , (1/x.m)*u.Fx);
    sh.setODE( 'dy'    , (1/x.m)*u.Fy);
    sh.setODE( 'dz'    , (1/x.m)*u.Fz - c.g);
    sh.setODE( 'phi'   , x.p + x.q*sin(x.phi)*tan(x.theta) + x.r*cos(x.phi)*tan(x.theta) );
    sh.setODE( 'theta' ,       x.q*cos(x.phi)              - x.r*sin(x.phi)              );
    sh.setODE( 'psi'   ,       x.q*sin(x.phi)/cos(x.theta) + x.r*cos(x.phi)/cos(x.theta) );
    sh.setODE( 'p'     , (1/Ix)*(u.L + (Iy-Iz)*x.q*x.r) );
    sh.setODE( 'q'     , (1/Iy)*(u.M + (Iz-Ix)*x.r*x.p) );
    sh.setODE( 'r'     , (1/Iz)*(u.N + (Ix-Iy)*x.p*x.q) );
    sh.setODE( 'm'     , -sqrt((u.Fx)^2 + (u.Fy)^2 + (u.Fz)^2 + (u.L)^2 + (u.M)^2 + (u.N)^2) / (c.Isp*c.g0));


end

function landerpathcosts(ch,x,u,~)
    
    % Cost Function (thrust magnitude)
    ch.add( sqrt((u.Fx)^2 + (u.Fy)^2 + (u.Fz)^2 + (u.L)^2 + (u.M)^2 + (u.N)^2 + 1.0) );
%     ch.add(u.Fx^2);
%     ch.add(u.Fy^2);
%     ch.add(u.Fz^2);
%     ch.add(u.L^2);
%     ch.add(u.M^2);
%     ch.add(u.N^2);
    % Time
%     ch.add(1);

end

function landerterminalcosts(ch,x,c)
    
%     ch.add(5e5*(norm([x.x,x.y] - c.objective)));

end


function landergridconstraints(ch,k,K,x,c)
    %Constrain y to being above surface and  final y to surface
% %     if k<K
% %         ch.add(x.y, '>=', c.surfFunc(x.x));
% %     end
% %     
% %     if k==K
% %          ch.add(x.y, '==', c.surfFunc(x.x));
% %          ch.add(x.x, '>=', -100);
% %          ch.add(x.x, '<=',  100);
% %     end
end



