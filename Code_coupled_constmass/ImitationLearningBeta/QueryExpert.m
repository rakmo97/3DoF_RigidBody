function proxyOut = QueryExpert(ICs)
    proxyOut = 0; % Dummy variable because python requires at least one output
    

    fid = fopen( 'ExpertTracking.txt', 'wt' );

    root_folder = 'E:/Research_Data/3DoF_RigidBody/Code_coupled_constmass/';
    saveout = [root_folder,'ImitationLearningBeta/QueriedTrajectories/d',datestr(now,'yyyymmdd_HHoMM'),'_genTrajs','.mat'];

    % Number of ICs to query OpenOCL for
    numICs = size(ICs,1);

    % Target State [x,y,phi,dx,dy,dphi,m]
    target = [0,0,0,0,0,0,0];
    objective = [0,0];
    
    % Preallocations
    N = 100;
    Jout = zeros(numICs,3);
    stateOut = zeros(N,7,numICs);
    ctrlOut = zeros(N,2,numICs);
    stateFinal = zeros(numICs,6);


    % Loop though
    for i = 1:numICs

        fprintf(fid, "On loop %d of %d!\n", i, numICs);
        
        
        % OpenOCL setup on first loop
        if i == 1
            run('../../OpenOCL/ocl.m');
        end


        % parameters

        conf = struct;
        conf.g = 9.81/6; % m/s2 
        conf.g0 = 9.81; % m/s2
        conf.r = 0.25;
        conf.Isp = 300;
        conf.m = 500;




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
        
%         fprintf(fid, "On state %d is %f!", i, ICs(i,:));

        
        % Set Initial Conditions
        solver.setInitialBounds( 'x'   ,   ICs(i,1)   );
        solver.setInitialBounds( 'y'   ,   ICs(i,2)   );
        solver.setInitialBounds( 'phi' ,   min(max(ICs(i,3),-pi/4),pi/4)   );
        solver.setInitialBounds( 'dx'  ,   ICs(i,4)   );
        solver.setInitialBounds( 'dy'  ,   ICs(i,5)   );
        solver.setInitialBounds( 'dphi',   ICs(i,6)   );


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
        
        if ~strcmp(solver.solver.stats.return_status,'Solve_Succeeded')
            
            Jout(i,:) = [NaN(1,3)];
            stateOut(:,:,i) = [NaN(N,7,1)];
            ctrlOut(:,:,i) = [NaN(N,2,1)];
            stateFinal(i,:) = [NaN(1,6)];

            continue
            
        end
        
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

        xa     = solution.states.x.value;
        ya     = solution.states.y.value;
        phia   = solution.states.phi.value;
        dxa    = solution.states.dx.value;
        dya    = solution.states.dy.value;
        dphia  = solution.states.dphi.value;


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
        stateOut(:,:,i) = [tc',x',y',phi',dx',dy',dphi'];
        ctrlOut(:,:,i) = [Fx',Fy'];
        stateFinal(i,:) = [xa(end),ya(end),phia(end),dxa(end),dya(end),dphia(end)];



    end

    % Get rid of nan rows
    nanidxs = find(isnan(Jout(:,1)));
    Jout(nanidxs,:) = [];
    stateOut(:,:,nanidxs) = [];
    ctrlOut(:,:,nanidxs) = [];
    stateFinal(nanidxs,:) = [];
    
    fprintf(fid, "Saving Optimized Trajectories!\n");

    % Save files
    save(saveout,'Jout','stateOut','ctrlOut','stateFinal');

    figure;
    hold on
    for i = 1:size(stateOut,3)
       plot(stateOut(:,2,i),stateOut(:,3,i)); 
    end
    title(['Plots of Queried Trajectories'])
    xlabel('X [m]')
    ylabel('Y [m]')
    saveas(gcf,[root_folder,'ImitationLearningBeta/QueriedTrajectoriesPlots/',saveout(end-27:end-4),'.png'])

    %% Aggregate Data
    fprintf(fid, "Aggregating Data!\n");

    
    % Setup Directory
    directory = [root_folder,'TrajectoryGeneration/ToOrigin_Trajectories'];
    addpath(directory);
    datadir = dir(directory);
    filenames = {datadir.name};
    datafiles = filenames(3:end);

    directory = [root_folder,'ImitationLearningBeta/QueriedTrajectories'];
    addpath(directory);
    datadir = dir(directory);
    filenames = {datadir.name};
    datafiles2 = filenames(3:end);

    datafiles =  {[datafiles,datafiles2]};
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
%     Jout = zeros(numdata/100,3);
%     ctrlOut = zeros(100,2,numdata/100);
%     stateOut = zeros(100,7,numdata/100);
    Jout = [];
    ctrlOut = [];
    stateOut = [];

    count = 1;
    for i = 1:length(datafiles)

        d = load(datafiles{i});

        disp(['Extracting datafile ',num2str(i),' of ',num2str(length(datafiles))]);

        lastidx = find(d.Jout(:,1)==0,1) - 1; % Find last index
        if isempty(lastidx)
            lastidx = size(d.Jout,1);
        end

        Jout = cat(1,Jout,d.Jout(1:lastidx,:));
        ctrlOut = cat(3,ctrlOut,d.ctrlOut(:,:,1:lastidx));
        stateOut = cat(3,stateOut,d.stateOut(:,:,1:lastidx));
    end
    disp('Done extracting')

    fprintf(fid, "Saving Aggregated Data!\n");

    % Save data to .mat file
    disp('Saving data to mat file')

    save([root_folder,'ImitationLearningBeta/','ANN2_aggregated_data.mat'],'stateOut','ctrlOut','Jout');

    disp('Saved')

    fprintf(fid, "Completed Expert Query!\n");
    fclose(fid);

end



%% Solver Functions
function landervarsfun(sh, c)


    % Define States
    sh.addState('x');
    sh.addState('y', 'lb', 0);
    sh.addState('phi', 'lb', -pi/4, 'ub', pi/4);
    sh.addState('dx');
    sh.addState('dy');
    sh.addState('dphi');

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
    sh.addParameter('m');

end

function landereqfun(sh,x,~,u,c) 

    
    J = c.m;

    
    % Equations of Motion
    sh.setODE( 'x'   , x.dx);
    sh.setODE( 'y'   , x.dy);
    sh.setODE( 'phi' , x.dphi);
    sh.setODE( 'dx'  , (1/c.m)*(u.Fx*cos(x.phi) - u.Fy*sin(x.phi)));
    sh.setODE( 'dy'  , (1/c.m)*(u.Fx*sin(x.phi) + u.Fy*cos(x.phi)) - c.g);
    sh.setODE( 'dphi', (1/J)*(c.r*u.Fx));


end

function landerpathcosts(ch,x,u,~)
    
    % Cost Function (thrust magnitude)
%     ch.add(u.Fx^2);
%     ch.add(u.Fy^2);
    ch.add(sqrt((u.Fx)^2 + (u.Fy)^2 + 1.0));


end




