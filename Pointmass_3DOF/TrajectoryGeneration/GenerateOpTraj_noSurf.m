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
nTrajs = 1; % Number of trajectories to generate
plotting = 1; % Plot things or no?
saveout = ['d',datestr(now,'yyyymmdd_HHoMM'),'_genTrajs','.mat'];

% Lower and upper values for random initial conditions
% [x,y,phi,dx,dy,dphi,m]
lower = [-1000, 1000, -1000, -0.5, -0.5, -0.5]';
upper = [ 1000, 1500,  1000,  0.5,  0.0,  0.5]';

% Target State [x,y,phi,dx,dy,dphi,m]
target = [0,0,0,0,0,0,0];

% Preallocations
N = 10;
surfFunctionOut = cell(nTrajs,1);
objectiveOut = zeros(nTrajs,2);
Jout = zeros(nTrajs,1);
stateOut = zeros(N,8,nTrajs);
ctrlOut = zeros(N,3,nTrajs);
runTimeOut = zeros(nTrajs,1);
stateFinal = zeros(nTrajs,7);

for i = 1:nTrajs
 
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
            conf.r = 1;
            conf.Isp = 300;
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

            solver = ocl.Problem([], varsfun, daefun, pathcosts, 'N',N);

            %% Populate Solver Settings
            % Parameters
            solver.setParameter('g'    , conf.g    );
            solver.setParameter('g0'   , conf.g0   );
            solver.setParameter('r'    , conf.r    );

            % Generate Initial Conditions and Target [x,y,phi,dx,dy,dphi,m]
            x0 = [0,0,0,-20,-1,0,500];
            r = lower+(upper-lower).*rand(6,1);
%             r = 1.0e+03*[1,1.3,0,0,0,0];
%             r = [1000,1000,1000,-15,-15,-15]';
              r = [630, -745, 1450, 0.5, -0.2, -0.4];
%             r = [1000,1000,0];
            % Set Initial Conditions
            solver.setInitialBounds( 'x'   ,   r(1)   );
            solver.setInitialBounds( 'y'   ,   r(2)   );
            solver.setInitialBounds( 'z' ,   r(3)   );
            solver.setInitialBounds( 'dx'  ,   r(4)   );
            solver.setInitialBounds( 'dy'  ,   r(5)   );
            solver.setInitialBounds( 'dz',   r(6)   );
            solver.setInitialBounds( 'm'   ,   x0(7)   );


            % Set Target State
            solver.setEndBounds( 'x' ,    target(1) );
            solver.setEndBounds( 'y' ,    target(2) );
            solver.setEndBounds( 'z' ,  target(3) );
            solver.setEndBounds( 'dx'  ,  target(4) );
            solver.setEndBounds( 'dy'  ,  target(5) );
            solver.setEndBounds( 'dz',  target(6) );



    
    
            %% Run Solver
            disp(['Starting trajectory ',num2str(i),' of ',num2str(nTrajs),'....'])

            tic
            initialGuess    = solver.getInitialGuess();
            [solution,times] = solver.solve(initialGuess);
            timeToRun = toc;
            
            if ~strcmp(solver.solver.stats.return_status,'Solve_Succeeded')
                error('Optimal Solution Not Found, Retrying...')
            end
            
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
            m  = solution.states.m.value;

            xa     = solution.states.x.value;
            ya     = solution.states.y.value;
            za   = solution.states.z.value;
            dxa    = solution.states.dx.value;
            dya    = solution.states.dy.value;
            dza  = solution.states.dz.value;
            ma  = solution.states.m.value;


            % Pull out controls
            Fx = solution.controls.Fx.value;
            Fy = solution.controls.Fy.value;
            Fz = solution.controls.Fz.value;


            % Define indexes of states that align with control values
            idxs = 1:ratio:length(ts)-1;

            % Separate states by available controls
            x     = x(idxs);
            y     = y(idxs);
            z   = z(idxs);
            dx    = dx(idxs);
            dy    = dy(idxs);
            dz  = dz(idxs);
            m     = m(idxs);


            % Calculate Costs
            L_F = Fx.^2 + Fy.^2 + Fz.^2;
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
            stateOut(:,:,i) = [tc',x',y',z',dx',dy',dz',m'];
            ctrlOut(:,:,i) = [Fx',Fy',Fz'];
            runTimeOut(i) = timeToRun;
            stateFinal(i,:) = [xa(end),ya(end),za(end),dxa(end),dya(end),dza(end),ma(end)];
            
            if plotting
                % Plot x,y,z trajectory
                figure(1);
                plot3(x(1),y(1),z(1),'rx','MarkerSize',10)
                hold on
                grid on
                plot3(solution.states.x.value,...
                   solution.states.y.value,...
                   solution.states.z.value,...
                   'Color','b','LineWidth',1.5);
                plot3(xa(end),ya(end),za(end),'bo','MarkerSize',10)
        %         plot(objective(1),objective(2),'c+','MarkerSize',10)
        %         plot(gridPoints(:,1),gridPoints(:,2),'.')
        %         plot(linspace(-100,100),surfFunction(linspace(-100,100)))
                xlabel('x[m]');ylabel('y[m]');; zlabel('z [m]')
                legend('Starting Point','Trajectory','Ending Point','Objective','Surface','location','best')


                % Plot thrust profiles
                figure(2);
                subplot(3,1,1)
                plot(tc,Fx,'g')
                hold on
                title('Controls')
                ylabel('F_x [N]')
                subplot(3,1,2)
                plot(tc,Fy,'b')
                hold on
                ylabel('F_y [N]')                
                subplot(3,1,3)
                plot(tc,Fz,'b')
                hold on
                ylabel('F_z [N]')


                figure(3);
                subplot(2,2,1)
                hold on
                plot(tc,x,'g')
                title('Position vs Time')
                ylabel('x [m]')
                subplot(2,2,2)
                plot(tc,y,'b')
                hold on
                ylabel('y [m]')
                subplot(2,2,3)
                plot(tc,z,'b')
                hold on
                ylabel('z [m]')
                subplot(2,2,4)
                plot(tc,m,'b')
                hold on
                ylabel('m [kg]')

        % 
        %         figure(4);
        %         subplot(3,1,1)
        %         hold on
        %         plot(tc,rad2deg(phi),'g')
        %         title('Euler Angles vs Time')
        %         ylabel('\phi [deg]')
        %         subplot(3,1,2)
        %         plot(tc,rad2deg(theta),'b')
        %         ylabel('\theta [deg]')
        %         subplot(3,1,3)
        %         plot(tc,rad2deg(psi),'r')
        %         ylabel('\psi [deg]')
        %         hold off
        %         xlabel('Time [s]')
        % 
        %         figure(5)
        %         subplot(3,1,1)
        %         hold on
        %         plot(tc,u,'g')
        %         title('v-body vs Time')
        %         ylabel('u [m/s]')
        %         subplot(3,1,2)
        %         plot(tc,v,'b')
        %         ylabel('v [m/s]')
        %         subplot(3,1,3)
        %         plot(tc,w,'r')
        %         ylabel('w [m/s]')
        %         hold off
        %         xlabel('Time [s]')
        % 
        %         figure(6);
        %         subplot(3,1,1)
        %         hold on
        %         plot(tc,rad2deg(p),'g')
        %         title('Body rates vs Time')
        %         ylabel('p [deg/s]')
        %         subplot(3,1,2)
        %         plot(tc,rad2deg(q),'b')
        %         ylabel('q [deg/s]')
        %         subplot(3,1,3)
        %         plot(tc,rad2deg(r),'r')
        %         ylabel('r [deg/s]')
        %         hold off
        %         xlabel('Time [s]')
            end
%             
        catch
            disp('Optimal Solution Not Found, Retrying...');
            err_count = err_count+1;
        end
        whilecount = whilecount+1;
    end


    disp(['Finished trajectory ',num2str(i),' of ',num2str(nTrajs)])
    
    
    
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
    sh.addState('m');

    Fmax = 15000;
    % Define Controls
    sh.addControl('Fx', 'lb', -Fmax, 'ub', Fmax);  % Force [N]
    sh.addControl('Fy', 'lb', -Fmax, 'ub', Fmax);  % Force [N]
    sh.addControl('Fz', 'lb', 0, 'ub', Fmax);  % Force [N]


    % System Parameters
    sh.addParameter('g');
    sh.addParameter('g0');
    sh.addParameter('r');
    sh.addParameter('Isp')
    sh.addParameter('objective');
    sh.addParameter('surfFunc');

end

function landereqfun(sh,x,~,u,c) % https://charlestytler.com/quadcopter-equations-motion/


    

    
    % Equations of Motion
    sh.setODE( 'x'   , x.dx);
    sh.setODE( 'y'   , x.dy);
    sh.setODE( 'z'   , x.dz);
    sh.setODE( 'dx'  , (1/x.m)*u.Fx);
    sh.setODE( 'dy'  , (1/x.m)*u.Fy);
    sh.setODE( 'dz'  , (1/x.m)*u.Fz - c.g);
    sh.setODE( 'm'   , -sqrt((u.Fx)^2 + (u.Fy)^2 + (u.Fz)^2) / (c.Isp*c.g0));


end

function landerpathcosts(ch,x,u,~)
    
    % Cost Function (thrust magnitude)
    ch.add(sqrt((u.Fx)^2 + (u.Fy)^2 + (u.Fz)^2));
%     ch.add(u.Fx^2);
%     ch.add(u.Fy^2);
%     ch.add(u.Fz^2);
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



