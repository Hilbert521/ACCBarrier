function [OUT] = Bounds_Computation_Synthesis(Orig, Target, Known_Bounds, Domain, Mode_L)

addpath(genpath('SDPT3-4.0'))
addpath(genpath('SOSTOOLS.303'))
poolobj = gcp('nocreate');
delete(poolobj);
%clear all; clc; close all;
x1 = sym('x1', 'real');
x2 = sym('x2', 'real');
z = sym('z', 'real');

Mode = Mode_L(1);

%Over and under-approximation computation


H_thresh = 0.000;

q1x = Orig(3) - Orig(1);
q1y = Orig(4) - Orig(2);
q2x = Target(3) - Target(1);
q2y = Target(4) - Target(2);
Domx = Domain(3) - Domain(1);
Domy = Domain(4) - Domain(2);

q1_center = [mean([Orig(3), Orig(1)]), mean([Orig(2),Orig(4)])];
q2_center = [mean([Target(3), Target(1)]), mean([Target(2),Target(4)])];
Dom_center = [mean([Domain(3), Domain(1)]), mean([Domain(2),Domain(4)])];


load('pqfile.mat')
load('pqfile_under.mat')


%Overapproximation of q1

P_q1over = P_over;

for i = 0:p_over
    for j = 0:p_over
        P_q1over(i+1, j+1) = (1/q1x)^(2*p_over-i-j)* (1/q1y)^(i+j) *P_q1over(i+1, j+1);
    end
end

X = sym([]);
for i = 0:p_over
    X(i+1,1) = (x1-q1_center(1))^(p_over-i)*(x2 - q1_center(2))^i;
end

polyq1_over = expand(b_over - X'*P_q1over*X);




%Overapproximation of q2

P_q2over = P_over;

for i = 0:p_over
    for j = 0:p_over
        P_q2over(i+1, j+1) = (1/q2x)^(2*p_over-i-j)* (1/q2y)^(i+j) *P_q2over(i+1, j+1);
    end
end

X = sym([]);
for i = 0:p_over
    X(i+1,1) = (x1-q2_center(1))^(p_over-i)*(x2 - q2_center(2))^i;
end

polyq2_over = expand(b_over - X'*P_q2over*X);



%Underapprox. of q2/Over approximation of outside of q2

P_q2under = P_under;

for i = 0:p_under
    for j = 0:p_under
        P_q2under(i+1, j+1) = (1/q2x)^(2*p_under-i-j)* (1/q2y)^(i+j) *P_q2under(i+1, j+1);
    end
end


X = sym([]);
for i = 0:p_under
    X(i+1,1) = (x1-q2_center(1))^(p_under-i)*(x2 - q2_center(2))^i;
end

polyq2_out =  expand(-b_under + X'*P_q2under*X);


%Overapprox. of domain

P_domover = P_over;

for i = 0:p_over
    for j = 0:p_over
        P_domover(i+1, j+1) = (1/Domx)^(2*p_over-i-j)* (1/Domy)^(i+j) *P_domover(i+1, j+1);
    end
end

X = sym([]);
for i = 0:p_over
    X(i+1,1) = (x1-Dom_center(1))^(p_over-i)*(x2 - Dom_center(2))^i;
end

poly_dom = expand(b_over - X'*P_domover*X);



% Bounds Computations

solver_opt.solver = 'sdpt3';


Bound_Up = Known_Bounds(2);
Bound_Low = Known_Bounds(1);

gx = 0.18;              % Range of noise values             
deg = 6;


% fx = [  x1*x2; ...
%         x2*x1^2 ; ...
%             ];

if Mode == 0
fx = [  6.0*(x1^3)*x2; ...    %Define function (SEEMS TO WORK FOR DEGREE 10)
        0.3*x2*x1 ; ...
            ];
else
fx = [  7.0*(x1^3)*x2; ...    %Define function (SEEMS TO WORK FOR DEGREE 10)
        0.2*x2*x1 ; ...
            ];    
end
fx1 = fx(1);  
fx2 = fx(2)+z;

%we compute an upper bound on the probability of making a transition from q1 to q2        

clear x1 x2 z      
   
tic
 
z = sym('z', 'real');
betasym = sym('betasym', 'real');
x1 = sym('x1', 'real');
x2 = sym('x2', 'real');

if exist('prog') == 1
    clear prog;
end


    EXP = 0;
    prog = sosprogram([x1, x2], betasym);
    Zmon = monomials([x1, x2], 0:deg);
    [prog, B0] = sospolyvar(prog, Zmon, 'wscoeff');
    [prog, sig_u] = sospolyvar(prog, Zmon);
    [prog, sig_o2] = sospolyvar(prog, Zmon);

    prog = sosineq(prog, sig_u);
    prog = sosineq(prog, sig_o2);

    prog = sosineq(prog, B0);       
    prog = sosineq(prog, Bound_Up - betasym); % Upper bound on the objective function
    prog = sosineq(prog, -Bound_Low +betasym); % Lower bound on the objective function
    prog = sosineq(prog, B0 - expand(sig_u*polyq2_over) - 1.0); %B1 is greater than 1 inside the overapproximation of q2
    stdvar = gx;

    Bsub = expand(subs(B0, [x1,x2], [fx1, fx2]));
    termlist = children(Bsub);        

    for ii = 1:length(termlist)
        zcount = 0;
        x1count = 0;
        x2count = 0;

        % Initialize expectation as zero value
        EXPz = 0;

        factoredterm = factor(termlist(ii));
        % Count power of each variable in monomial
        for jj = 1:length(factoredterm)

            % Count the power of the "noise" term
            if isequaln(factoredterm(jj),z)
                zcount = zcount + 1;
            end

            % Count nubmer of states
            if isequaln(factoredterm(jj),x1)
                x1count = x1count + 1;
            end

            if isequaln(factoredterm(jj),x2)
                x2count = x2count + 1;
            end
        end

        % Check to see if symbolic term has no noise variables
        % May or may not have state variables
        if zcount == 0
            EXPz = EXPz + termlist(ii);
        end

        % Check if there are "noise" terms but not x variables
        if zcount > 0 && x1count == 0 && x2count == 0
            if mod(zcount,2) == 1
                EXPz = 0;
            elseif mod(zcount,2) == 0
                EXPz = prod(factoredterm(find(factoredterm~=z)))*prod([1:2:zcount])*stdvar^zcount;
            end
        end

        % Check if there are "noise" terms and x variables
        if zcount > 0 && (x1count > 0 || x2count > 0)
            if mod(zcount,2) == 1
                EXPz = 0;
            elseif mod(zcount,2) == 0
                EXPz = prod(factoredterm(find(factoredterm~=z)))*prod([1:2:zcount])*stdvar^zcount;
            end
        end

        EXP = EXP + EXPz;
    end


    prog = sosineq(prog, -EXP + betasym - expand(sig_o2*polyq1_over)); % Constraints on the expectation
    objfunc = betasym;
    prog = sossetobj(prog, objfunc);

    %Solve the program

    prog = sossolve(prog, solver_opt);

    if prog.solinfo.info.dinf == 1 || prog.solinfo.info.pinf == 1
        list_prob = Bound_Up;
    else    
        B0polys = sosgetsol(prog, B0);
        betaval = double(sosgetsol(prog, betasym));
        probvalue = betaval;
        list_prob = probvalue;
    end






toc
       
H = list_prob;   
               
if H < H_thresh
    H = 0;
end

if H > Known_Bounds(2)
    H = Known_Bounds(2);
else    
    Known_Bounds(2) = H;
end
        

if H == 0
    L = 0;
    Known_Bounds(1) = 0;
else    
        
    
   
 Bound_Up = 1 - Known_Bounds(1);
 Bound_Low = 1 - Known_Bounds(2);
     
         
 
     
 %we compute a lower bound on the probability of making a transition from q1 to q2
 
 
 
 z = sym('z', 'real');
 betasym = sym('betasym', 'real');
 x1 = sym('x1', 'real');
 x2 = sym('x2', 'real');
 
 
 if exist('prog') == 1
     clear prog;
 end
 
 
 EXP = 0;
 prog = sosprogram([x1, x2], betasym);
 Zmon = monomials([x1, x2], [0:deg]);
 [prog, B0] = sospolyvar(prog, Zmon, 'wscoeff');
 [prog, sig_u] = sospolyvar(prog, Zmon);
 [prog, sig_o2] = sospolyvar(prog, Zmon);
 prog = sosineq(prog, sig_u);
 prog = sosineq(prog, sig_o2);
 prog = sosineq(prog, B0); %B0 is positive on the (over-approximation of the) domain
 prog = sosineq(prog, Bound_Up -betasym); % Upper bound on the objective function
 prog = sosineq(prog, -Bound_Low + betasym); % Lower bound on the objective function
 prog = sosineq(prog, B0 - sig_u*polyq2_out - 1); %B1 is greater than 1 outside of the underapproximation of q2
 stdvar = gx;
 
 Bsub = expand(subs(B0, [x1,x2], [fx1, fx2]));
 termlist = children(Bsub);
 % NEED TO COMPUTE EXP (Expectation term) as function of B1
 %
 % The following equations are for computing the Expected value
 for ii = 1:length(termlist)
     zcount = 0;
     x1count = 0;
     x2count = 0;
 
     % Initialize expectation as zero value
     EXPz = 0;
 
     factoredterm = factor(termlist(ii));
     % Count power of each variable in monomial
     for jj = 1:length(factoredterm)
 
         % Count the power of the "noise" term
         if isequaln(factoredterm(jj),z)
             zcount = zcount + 1;
         end
 
         % Count nubmer of states
         if isequaln(factoredterm(jj),x1)
             x1count = x1count + 1;
         end
 
         if isequaln(factoredterm(jj),x2)
             x2count = x2count + 1;
         end
     end
 
     % Check to see if symbolic term has no noise variables
     % May or may not have state variables
     if zcount == 0
         EXPz = EXPz + termlist(ii);
     end
 
     % Check if there are "noise" terms but not x variables
     if zcount > 0 && x1count == 0 && x2count == 0
         if mod(zcount,2) == 1
             EXPz = 0;
         elseif mod(zcount,2) == 0
             EXPz = prod(factoredterm(find(factoredterm~=z)))*prod([1:2:zcount])*stdvar^zcount;
         end
     end
 
     % Check if there are "noise" terms and x variables
     if zcount > 0 && (x1count > 0 || x2count > 0)
         if mod(zcount,2) == 1
             EXPz = 0;
         elseif mod(zcount,2) == 0
             EXPz = prod(factoredterm(find(factoredterm~=z)))*prod([1:2:zcount])*stdvar^zcount;
         end
     end
 
     EXP = EXP + EXPz;
 end
 
 
 prog = sosineq(prog, -EXP + betasym - expand(sig_o2*polyq1_over)); % Constraints on the expectation
 objfunc = betasym;
 prog = sossetobj(prog, objfunc);
 
 %Solve the program
 
 prog = sossolve(prog, solver_opt);
 if prog.solinfo.info.dinf == 1 || prog.solinfo.info.pinf == 1
     list_prob = Known_Bounds(1);
 else
     betaval = double(sosgetsol(prog, betasym));
     probvalue = betaval;
     list_prob = 1-probvalue;
 end
 
 
 
 
 
 end
 
 
 
 L = list_prob;
 Known_Bounds(1) = L;




poolobj = gcp('nocreate');
delete(poolobj);

Known_Bounds_o = Known_Bounds;

OUT = [H,L,Known_Bounds_o];

end
