function proxyOut = QueryExpert(ICs)
    proxyOut = 0; % Dummy variable because python requires at least one output
    

    fid = fopen( 'ExpertTracking.txt', 'wt' );

    saveout = ['QueriedTrajectories/d',datestr(now,'yyyymmdd_HHoMM'),'_genTrajs','.mat'];

    % Number of ICs to query OpenOCL for
    numICs = size(ICs,1);

    % Target State [x,y,phi,dx,dy,dphi,m]
    target = [0,0,0,0,0,0,0];
    objective = [0,0];
    
    % Preallocations
    N = 100;
    Jout = zeros(numICs,3);
    stateOut = zeros(N,8,numICs);
    ctrlOut = zeros(N,2,numICs);
    stateFinal = zeros(numICs,7);
    solverMessages = cell(numICs,1);
    bad_idxs = [];
    
    % Loop though
    for i = 1:numICs

        fprintf(fid, "On loop %d of %d!\n", i, numICs);
        
        
        % OpenOCL setup on first loop
        if i == 1
            run('../OpenOCL-7.07/ocl.m');
        end



        % parameters
        conf = struct;
        conf.g = 9.81/6; % m/s2
        conf.g0 = 9.81; % m/s2
        conf.r = 1;
        conf.Isp = 300;



        % Setup Solver
        varsfun    = @(sh) landervarsfun(sh, conf);
        daefun     = @(sh,x,z,u,p) landereqfun(sh,x,z,u,conf);
        pathcosts  = @(ch,x,~,u,~) landerpathcosts(ch,x,u,conf);

        solver = ocl.Problem([], varsfun, daefun, pathcosts, 'N',N);


        %Populate Solver Settings
        % Parameters
        solver.setParameter('g'    , conf.g    );
        solver.setParameter('g0'   , conf.g0   );
        solver.setParameter('r'    , conf.r    );

        % Set Initial Conditions
        solver.setInitialBounds( 'x'   ,   ICs(i,1)   );
        solver.setInitialBounds( 'y'   ,   ICs(i,2)   );
        solver.setInitialBounds( 'phi' ,   min(max(ICs(i,3),-pi/3),pi/3)   );
        solver.setInitialBounds( 'dx'  ,   ICs(i,4)   );
        solver.setInitialBounds( 'dy'  ,   ICs(i,5)   );
        solver.setInitialBounds( 'dphi',   ICs(i,6)   );
        solver.setInitialBounds( 'm'   ,   ICs(i,7)   );


        % Set Target State
        solver.setEndBounds( 'x' ,    target(1) );
        solver.setEndBounds( 'y' ,    target(2) );
        solver.setEndBounds( 'phi' ,  target(3) );
        solver.setEndBounds( 'dx'  ,  target(4) );
        solver.setEndBounds( 'dy'  ,  target(5) );
        solver.setEndBounds( 'dphi',  target(6) );

        % Run Solver
        initialGuess    = solver.getInitialGuess();
        [solution,times] = solver.solve(initialGuess);
        solverMessages{i} = solver.solver.stats.return_status;

        % Process solutions
        % Grab Times
        ts = times.states.value;
        tc = times.controls.value;
        ratio = (length(ts)-1)/length(tc); % Ratio of ctrl times to state times

        % Pull out states
        x     = solution.states.x.value;
        y     = solution.states.y.value;
        phi   = solution.states.phi.value;
        dx    = solution.states.dx.value;
        dy    = solution.states.dy.value;
        dphi  = solution.states.dphi.value;
        m  = solution.states.m.value;

        xa     = solution.states.x.value;
        ya     = solution.states.y.value;
        phia   = solution.states.phi.value;
        dxa    = solution.states.dx.value;
        dya    = solution.states.dy.value;
        dphia  = solution.states.dphi.value;
        ma  = solution.states.m.value;


        % Pull out controls
        Fx = solution.controls.Fx.value;
        Fy = solution.controls.Fy.value;


        % Define indexes of states that align with control values
        idxs = 1:ratio:length(ts)-1;

        % Separate states by available controls
        x     = x(idxs);
        y     = y(idxs);
        phi   = phi(idxs);
        dx    = dx(idxs);
        dy    = dy(idxs);
        dphi  = dphi(idxs);
        m     = m(idxs);


        % Calculate Costs
        L_F = Fx.^2 + Fy.^2;
        J_F = trapz(tc,L_F);


        J_path = J_F;
        J_term = norm([x(end),y(end)]-objective);
        J_total = J_path + J_term;

        %     disp(['Force min cost is ',num2str(J_F)])
        disp(['Path Cost is  ', num2str(J_path)])
        disp(['Term cost is ', num2str(J_term)])
        disp(['Total cost is ', num2str(J_total)])

        % Save off outputs
        Jout(i,:) = [J_path,J_term,J_total];
        stateOut(:,:,i) = [tc',x',y',phi',dx',dy',dphi',m'];
        ctrlOut(:,:,i) = [Fx',Fy'];
        stateFinal(i,:) = [xa(end),ya(end),phia(end),dxa(end),dya(end),dphia(end),ma(end)];

        % Detect if bad trajectory
        if strcmp(solverMessages{i},'Solve_Succeeded')
            bad_idxs = [bad_idxs; i];
        elseif any(ya < 0)
            bad_idxs = [bad_idxs; i];
        end

    end

    % Detecting and Removing bad trajectories
    fprintf(fid, "Detecting and Removing  %d bad trajectories...\n", length(bad_idxs));
    Jout(bad_idxs,:) = [];
    stateOut(:,:,bad_idxs) = [];
    ctrlOut(:,:,bad_idxs) = [];
    stateFinal(bad_idxs,:) = [];


    
    fprintf(fid, "Saving Optimized Trajectories!\n");
    % Save files
    save(saveout,'Jout','stateOut','ctrlOut','stateFinal');



    %% Aggregate Data
    fprintf(fid, "Aggregating Data!\n");

    
    % Setup Directory
    directory = '../TrajectoryGeneration/ToOrigin_Trajectories';
    addpath(directory);
    datadir = dir(directory);
    filenames = {datadir.name};
    datafiles = filenames(3:end);

    directory = '../TrajectoryGeneration/ToSurface_Trajectories';
    addpath(directory);
    datadir = dir(directory);
    filenames = {datadir.name};
    datafiles2 = filenames(3:end);

    directory = 'QueriedTrajectories';
    addpath(directory);
    datadir = dir(directory);
    filenames = {datadir.name};
    datafiles3 = filenames(3:end);

    datafiles =  {[datafiles,datafiles2,datafiles3]};
    datafiles = datafiles{1};


    % Pull data and format

    % Preallocation loop
    disp('Preallocating');
    numdata = 0;
    for i = 1:length(datafiles)
        d = load(datafiles{i});

        lastidx = find(d.Jout(:,1)==0,1) - 1; % Find last index
        if isempty(lastidx)
            lastidx = size(d.Jout,1);
        end

        numinthisfile = size(d.ctrlOut,1)*lastidx;
        numdata = numdata + numinthisfile;  

    end


    % Preallocate
    Xagg = zeros(numdata,7);
    tagg = zeros(numdata,3);
    times = zeros(numdata,1);

    count = 1;
    for i = 1:length(datafiles)

        d = load(datafiles{i});

        disp(['Extracting datafile ',num2str(i),' of ',num2str(length(datafiles))]);

        lastidx = find(d.Jout(:,1)==0,1) - 1; % Find last index
        if isempty(lastidx)
            lastidx = size(d.Jout,1);
        end


        for j = 1:lastidx

            for k = 1:100

                Xagg(count,:) = [d.stateFinal(j,1)-d.stateOut(k,2,j),... % x
                d.stateFinal(j,2)-d.stateOut(k,3,j),... % y
                d.stateFinal(j,3)-d.stateOut(k,4,j),... % phi
                d.stateFinal(j,4)-d.stateOut(k,5,j),... % dx
                d.stateFinal(j,5)-d.stateOut(k,6,j),... % dy
                d.stateFinal(j,6)-d.stateOut(k,7,j),... % dphi
                d.stateOut(k,8,j),...
                ];
            
                phi = d.stateOut(k,4,j);
                D = [cos(phi),-sin(phi);...
                     sin(phi), cos(phi);...
                     1       , 0    ];
            
                tagg(count,:) = (D*[d.ctrlOut(k,1,j),d.ctrlOut(k,2,j)]')';

                times(count) = d.stateOut(k,1,j);

%                 if(rem(count,100)==0)
%                     tagg(count,:) = [0,0];
%                 end

                count = count+1;
            end

        end
    end
    disp('Done extracting')

    fprintf(fid, "Saving Aggregated Data!\n");

    % Save data to .mat file
    disp('Saving data to mat file')

    save('ANN2_aggregated_data.mat','Xagg','tagg','times');

    disp('Saved')

    fprintf(fid, "Completed Expert Query!\n");
    fclose(fid);

end



%% Solver Functions
function landervarsfun(sh, c)


    % Define States
    sh.addState('x');
    sh.addState('y');
    sh.addState('phi', 'lb', -pi/3, 'ub', pi/3);
    sh.addState('dx');
    sh.addState('dy');
    sh.addState('dphi');
    sh.addState('m');

    Fmax = 15000;
    % Define Controls
    sh.addControl('Fx', 'lb', -Fmax, 'ub', Fmax);  % Force [N]
    sh.addControl('Fy', 'lb', 0, 'ub', Fmax);  % Force [N]


    % System Parameters
    sh.addParameter('g');
    sh.addParameter('g0');
    sh.addParameter('r');
    sh.addParameter('Isp')
    sh.addParameter('objective');
    sh.addParameter('surfFunc');

end

function landereqfun(sh,x,~,u,c) 

    
    J = c.r^2*x.m;

    
    % Equations of Motion
    sh.setODE( 'x'   , x.dx);
    sh.setODE( 'y'   , x.dy);
    sh.setODE( 'phi' , x.dphi);
    sh.setODE( 'dx'  , (1/x.m)*(u.Fx*cos(x.phi) - u.Fy*sin(x.phi)));
    sh.setODE( 'dy'  , (1/x.m)*(u.Fx*sin(x.phi) + u.Fy*cos(x.phi)) - c.g);
    sh.setODE( 'dphi', (1/J)*(c.r*u.Fx));
    sh.setODE( 'm'   , -sqrt((u.Fx)^2 + (u.Fy)^2) / (c.Isp*c.g0));


end

function landerpathcosts(ch,x,u,~)
    
    % Cost Function (thrust magnitude)
    ch.add(u.Fx^2);
    ch.add(u.Fy^2);

end




