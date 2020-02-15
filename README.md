# ACCBarrier

The files in this repository are used to produce the case studies of our paper titled “Interval-valued markov chain abstraction of stochastic systems using barrier functions”.

Running the files require Python 2.7, MATLAB and all files found in this repository, as well as the packages "cvx", "SDPT3-4.0" and "SOSTOOLS.303" which have to be located in the folder where these files are saved.

In order to generate the verification case study, run the "Barrier_Verification.py" file. In order to generate the synthesis case study, run the "Barrier_Synthesis.py" file. Note that we maximized the probability of the complement of specification phi to minimize the probability of phi.

The files "pqfile.mat" and pqfile_under.mat" contain under and over approximations of the unit squared center at the origin, which is used to approximate any state in the partition as detailed in the paper.

These files are unparallelized versions of the procedure. In order to speed up the runtime, feel free to parallelize the loops in the function "Probability_Interval_Computation_Barrier" in the file "Barrier_Verification_Functions.py", and the loops in the functions "BMDP_Probability_Interval_Computation_Barrier" in the file "Barrier_Synthesis_Functions.py".

If you have any questions or issues regarding the code, please email me at maxdutreix@gatech.edu.
