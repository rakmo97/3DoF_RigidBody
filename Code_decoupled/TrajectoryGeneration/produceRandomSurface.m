function [grid, target, surf] = produceRandomSurface()
while(true)

    %% Produce Surface
    lbound = -100;
    rbound = 100;
    nPoints = 20;
    x = linspace(lbound,rbound);
    xgrid = linspace(lbound,rbound,nPoints);

    noise_std = 0.5;

    Aa = -20;
    Ab = 20;
    Aran = Aa+(Ab-Aa).*rand(3,1);
    A1 = Aran(1);
    A2 = Aran(2);
    A3 = Aran(3);

    oa = -0.1;
    ob = 0.1;
    oran = oa+(ob-oa).*rand(3,1);
    o1 = oran(1);
    o2 = oran(2);
    o3 = oran(3);

    phia = -3;
    phib = 3;
    phiran = phia+(phib-phia).*rand(3,1);
    phi1 = phiran(1);
    phi2 = phiran(2);
    phi3 = phiran(3);

    surf = @(x) A1*sin(o1*x+phi1) + A2*sin(o2*x+phi2) + A3*sin(o3*x+phi3);

    y = surf(x);
    ygrid = surf(xgrid) + noise_std*randn(length(xgrid),1)';

    tya = -60;
    tyb = 60;
    txa = -100;
    txb = 100;
    targetx = txa+(txb-txa) .* rand;
    targety = tya+(tyb-tya) .* rand;
    if(targety <= surf(targetx) && targety>=1.5*min(ygrid))
        break;
    end





    % figure;
    % plot(x,y)
    % hold on
    % plot(xgrid,ygrid,'*')
    % plot(targetx,targety,'kx')
    % legend('Surface', 'Gridded Surface for Training','Objective')
    % xlabel('[meters]')
    % ylabel('[meters]')


    %% Generate Objective

    txa = -30;
    txb = 30;



    targetx = txa+(txb-txa) .* rand;

    tyb = surf(targetx);
    tya = tyb - 10;
    targety = tya+(tyb-tya) .* rand;


    grid(:,1) = xgrid;
    grid(:,2) = ygrid;
    target = [targetx,targety];


end


end