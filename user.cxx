//////////////////////////////////////////////////////////////////////////
////////////////        bryson_max_range.cxx         /////////////////////
//////////////////////////////////////////////////////////////////////////
////////////////           PSOPT  Example            /////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////// Title:         Bryson maximum range problem      ////////////////
//////// Last modified: 05 January 2009                   ////////////////
//////// Reference:     Bryson and Ho (1975)              ////////////////
//////// (See PSOPT handbook for full reference)          ////////////////
//////////////////////////////////////////////////////////////////////////
////////     Copyright (c) Victor M. Becerra, 2009        ////////////////
//////////////////////////////////////////////////////////////////////////
//////// This is part of the PSOPT software library, which////////////////
//////// is distributed under the terms of the GNU Lesser ////////////////
//////// General Public License (LGPL)                    ////////////////
//////////////////////////////////////////////////////////////////////////
#include <math.h>
#include "psopt.h"

//////////////////////////////////////////////////////////////////////////
///////////////////  Define the end point (Mayer) cost function //////////
//////////////////////////////////////////////////////////////////////////

adouble endpoint_cost(adouble* initial_states, adouble* final_states,
                      adouble* parameters,adouble& t0, adouble& tf,
                      adouble* xad, int iphase, Workspace* workspace)
{
//    adouble x = final_states[0];

//    return (-x);
    return 0.0;
}

//////////////////////////////////////////////////////////////////////////
///////////////////  Define the integrand (Lagrange) cost function  //////
//////////////////////////////////////////////////////////////////////////

adouble integrand_cost(adouble* states, adouble* controls, adouble* parameters,
                     adouble& time, adouble* xad, int iphase, Workspace* workspace)
{
    adouble Fx = controls[ 0 ];
    adouble Fy = controls[ 1 ];
    adouble Fz = controls[ 2 ];


    adouble L = sqrt(Fx*Fx + Fy*Fy + Fz*Fz);

    return  L;
}


//////////////////////////////////////////////////////////////////////////
///////////////////  Define the DAE's ////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

void dae(adouble* derivatives, adouble* path, adouble* states,
         adouble* controls, adouble* parameters, adouble& time,
         adouble* xad, int iphase, Workspace* workspace)
{
    adouble ddx, ddy, ddz, dm;

    double g = 9.81/6;
    double g0 = 9.81;
    double Isp = 300.0;

    adouble x  = states[ 0 ];
    adouble y  = states[ 1 ];
    adouble z  = states[ 2 ];
    adouble dx = states[ 3 ];
    adouble dy = states[ 4 ];
    adouble dz = states[ 5 ];
    adouble m  = states[ 6 ];

    adouble Fx = controls[ 0 ];
    adouble Fy = controls[ 1 ];
    adouble Fz = controls[ 2 ];


    ddx = Fx/m;
    ddy = (Fy/m) - g;
    ddz = Fz/m;
    dm  = -sqrt(Fx*Fx + Fy*Fy + Fz*Fz)/(Isp*g0);

    derivatives[ 0 ] = dx;
    derivatives[ 1 ] = dy;
    derivatives[ 2 ] = dz;
    derivatives[ 3 ] = ddx;
    derivatives[ 4 ] = ddy;
    derivatives[ 5 ] = ddz;
    derivatives[ 6 ] = dm;

}

////////////////////////////////////////////////////////////////////////////
///////////////////  Define the events function ////////////////////////////
////////////////////////////////////////////////////////////////////////////

void events(adouble* e, adouble* initial_states, adouble* final_states,
            adouble* parameters,adouble& t0, adouble& tf, adouble* xad,
            int iphase, Workspace* workspace)

{
   adouble x0  = initial_states[ 0 ];
   adouble y0  = initial_states[ 1 ];
   adouble z0  = initial_states[ 2 ];
   adouble dx0 = initial_states[ 3 ];
   adouble dy0 = initial_states[ 4 ];
   adouble dz0 = initial_states[ 5 ];
   adouble m0  = initial_states[ 6 ];

   adouble xf  = final_states[ 0 ];
   adouble yf  = final_states[ 1 ];
   adouble zf  = final_states[ 2 ];
   adouble dxf = final_states[ 3 ];
   adouble dyf = final_states[ 4 ];
   adouble dzf = final_states[ 5 ];

   e[ 0 ] =  x0; // Initial Conditions
   e[ 1 ] =  y0;
   e[ 2 ] =  z0;
   e[ 3 ] = dx0;
   e[ 4 ] = dy0;
   e[ 5 ] = dz0;
   e[ 6 ] =  m0;

   e[ 7 ] =  xf; // Final Conditions
   e[ 8 ] =  yf;
   e[ 9 ] =  zf;
   e[ 10] = dxf;
   e[ 11] = dyf;
   e[ 12] = dzf;

}



///////////////////////////////////////////////////////////////////////////
///////////////////  Define the phase linkages function ///////////////////
///////////////////////////////////////////////////////////////////////////

void linkages( adouble* linkages, adouble* xad, Workspace* workspace)
{
  // No linkages as this is a single phase problem
}



////////////////////////////////////////////////////////////////////////////
///////////////////  Define the main routine ///////////////////////////////
////////////////////////////////////////////////////////////////////////////

int main(void)
{

////////////////////////////////////////////////////////////////////////////
///////////////////  Declare key structures ////////////////////////////////
////////////////////////////////////////////////////////////////////////////

    Alg  algorithm;
    Sol  solution;
    Prob problem;

////////////////////////////////////////////////////////////////////////////
///////////////////  Register problem name  ////////////////////////////////
////////////////////////////////////////////////////////////////////////////

    problem.name        		= "3DOF Pointmass Lander";
    problem.outfilename                 = "pointmass3dof.txt";

////////////////////////////////////////////////////////////////////////////
////////////  Define problem level constants & do level 1 setup ////////////
////////////////////////////////////////////////////////////////////////////

    problem.nphases   			        = 1;
    problem.nlinkages                   = 0;

    psopt_level1_setup(problem);

/////////////////////////////////////////////////////////////////////////////
/////////   Define phase related information & do level 2 setup  ////////////
/////////////////////////////////////////////////////////////////////////////

    problem.phases(1).nstates   		= 7;
    problem.phases(1).ncontrols 		= 3;
    problem.phases(1).nevents   		= 13;
    problem.phases(1).npath     		= 0;
    problem.phases(1).nodes         <<   10;

    psopt_level2_setup(problem, algorithm);

////////////////////////////////////////////////////////////////////////////
///////////////////  Declare MatrixXd objects to store results //////////////
////////////////////////////////////////////////////////////////////////////

    MatrixXd x, u, t;
    MatrixXd lambda, H;

////////////////////////////////////////////////////////////////////////////
///////////////////  Enter problem bounds information //////////////////////
////////////////////////////////////////////////////////////////////////////

    double Fmax = 15000.0; // [N]

    double yL = 0.0;
    double FxL = -Fmax;
    double FxU =  Fmax;
    double FyL =  0.0;
    double FyU =  Fmax;
    double FzL = -Fmax;
    double FzU =  Fmax;

    double x0  = 1000.0; // Initial states
    double y0  = 1000.0;
    double z0  = 1000.0;
    double dx0 = 15.0;
    double dy0 = 0.0;
    double dz0 = 15.0;
    double m0  = 500.0;
    double xf  = 0.0; // Final states
    double yf  = 0.0;
    double zf  = 0.0;
    double dxf = 0.0;
    double dyf = 0.0;
    double dzf = 0.0;

    // State/Control constraints
    // problem.phases(1).bounds.lower.states(1) = yL;


    problem.phases(1).bounds.lower.controls(0) = FxL;
    problem.phases(1).bounds.lower.controls(1) = FyL;
    problem.phases(1).bounds.lower.controls(2) = FzL;
    problem.phases(1).bounds.upper.controls(0) = FxU;
    problem.phases(1).bounds.upper.controls(1) = FyU;
    problem.phases(1).bounds.upper.controls(2) = FzU;

    // Boundary Constraints
    problem.phases(1).bounds.lower.events(0) = x0;
    problem.phases(1).bounds.upper.events(0) = x0;

    problem.phases(1).bounds.lower.events(1) = y0;
    problem.phases(1).bounds.upper.events(1) = y0;

    problem.phases(1).bounds.lower.events(2) = z0;
    problem.phases(1).bounds.upper.events(2) = z0;

    problem.phases(1).bounds.lower.events(3) = dx0;
    problem.phases(1).bounds.upper.events(3) = dx0;

    problem.phases(1).bounds.lower.events(4) = dy0;
    problem.phases(1).bounds.upper.events(4) = dy0;

    problem.phases(1).bounds.lower.events(5) = dz0;
    problem.phases(1).bounds.upper.events(5) = dz0;

    problem.phases(1).bounds.lower.events(6) = m0;
    problem.phases(1).bounds.upper.events(6) = m0;

    problem.phases(1).bounds.lower.events(7) = xf;
    problem.phases(1).bounds.upper.events(7) = xf;

    problem.phases(1).bounds.lower.events(8) = yf;
    problem.phases(1).bounds.upper.events(8) = yf;

    problem.phases(1).bounds.lower.events(9) = zf;
    problem.phases(1).bounds.upper.events(9) = zf;

    problem.phases(1).bounds.lower.events(10) = dxf;
    problem.phases(1).bounds.upper.events(10) = dxf;

    problem.phases(1).bounds.lower.events(11) = dyf;
    problem.phases(1).bounds.upper.events(11) = dyf;

    problem.phases(1).bounds.lower.events(12) = dzf;
    problem.phases(1).bounds.upper.events(12) = dzf;

    // Time constraints
    problem.phases(1).bounds.lower.StartTime    = 0.0;
    problem.phases(1).bounds.upper.StartTime    = 0.0;

    problem.phases(1).bounds.lower.EndTime      = 30.0;
    problem.phases(1).bounds.upper.EndTime      = 100.0;



////////////////////////////////////////////////////////////////////////////
///////////////////  Register problem functions  ///////////////////////////
////////////////////////////////////////////////////////////////////////////


    problem.integrand_cost 	= &integrand_cost;
    problem.endpoint_cost 	= &endpoint_cost;
    problem.dae             = &dae;
    problem.events 		    = &events;
    problem.linkages		= &linkages;

////////////////////////////////////////////////////////////////////////////
///////////////////  Define & register initial guess ///////////////////////
////////////////////////////////////////////////////////////////////////////

    int nnodes    			            = problem.phases(1).nodes(0);
    int ncontrols                       = problem.phases(1).ncontrols;
    int nstates                         = problem.phases(1).nstates;

    MatrixXd x_guess    =  zeros(nstates,nnodes);

    x_guess.row(0)  =  x0*ones(1,nnodes);
    x_guess.row(1)  =  y0*ones(1,nnodes);
    x_guess.row(2)  =  z0*ones(1,nnodes);
    x_guess.row(3)  = dx0*ones(1,nnodes);
    x_guess.row(4)  = dy0*ones(1,nnodes);
    x_guess.row(5)  = dz0*ones(1,nnodes);
    x_guess.row(6)  =  m0*ones(1,nnodes);

    problem.phases(1).guess.controls       = zeros(ncontrols,nnodes);
    problem.phases(1).guess.states         = x_guess;
    problem.phases(1).guess.time           = linspace(0.0,60.0,nnodes);


////////////////////////////////////////////////////////////////////////////
///////////////////  Enter algorithm options  //////////////////////////////
////////////////////////////////////////////////////////////////////////////


    algorithm.nlp_iter_max                = 2500;
    algorithm.nlp_tolerance               = 1.e-4;
    algorithm.nlp_method                  = "IPOPT";
    // algorithm.scaling                     = "automatic";
    algorithm.derivatives                 = "automatic";
    algorithm.collocation_method          = "trapezoidal";
//    algorithm.defect_scaling = "jacobian-based";
    algorithm.ode_tolerance               = 1.e-6;



////////////////////////////////////////////////////////////////////////////
///////////////////  Now call PSOPT to solve the problem   /////////////////
////////////////////////////////////////////////////////////////////////////

    psopt(solution, problem, algorithm);

////////////////////////////////////////////////////////////////////////////
///////////  Extract relevant variables from solution structure   //////////
////////////////////////////////////////////////////////////////////////////


    x       = solution.get_states_in_phase(1);
    u       = solution.get_controls_in_phase(1);
    t       = solution.get_time_in_phase(1);
    lambda  = solution.get_dual_costates_in_phase(1);
    H       = solution.get_dual_hamiltonian_in_phase(1);


////////////////////////////////////////////////////////////////////////////
///////////  Save solution data to files if desired ////////////////////////
////////////////////////////////////////////////////////////////////////////

    Save(x, "x.dat");
    Save(u,"u.dat");
    Save(t,"t.dat");
    Save(lambda,"lambda.dat");
    Save(H,"H.dat");

////////////////////////////////////////////////////////////////////////////
///////////  Plot some results if desired (requires gnuplot) ///////////////
////////////////////////////////////////////////////////////////////////////

    plot(t,x,problem.name+": states", "time (s)", "states","x y z dx dy dz m");

    plot(t,u,problem.name+": controls","time (s)", "controls", "F_x F_y F_z");

    plot(t,x,problem.name+": states", "time (s)", "states","x y z dx dy dz m",
                             "pdf", "brymr_states.pdf");

    plot(t,u,problem.name+": controls","time (s)", "controls", "F_x F_y F_z",
                             "pdf", "brymr_controls.pdf");
}

////////////////////////////////////////////////////////////////////////////
///////////////////////      END OF FILE     ///////////////////////////////
////////////////////////////////////////////////////////////////////////////
