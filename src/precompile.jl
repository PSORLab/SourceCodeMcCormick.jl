function _precompile_()
    # Individual string kernels
    Base.precompile(SCMC_cadd_kernel, (String,String,Real)) # Time: 0.00023
    Base.precompile(SCMC_add_to_kernel, (String,String)) # Time: 0.004736
    Base.precompile(SCMC_negate_kernel, (String,String)) # Time: 0.003571
    Base.precompile(SCMC_exp_kernel, (String,String)) # Time: 0.035821
    Base.precompile(SCMC_log_kernel, (String,String)) # Time: 0.040379
    Base.precompile(SCMC_inv_kernel, (String,String)) # Time: 0.104374
    Base.precompile(SCMC_cmul_kernel, (String,String)) # Time: 0.000003
    Base.precompile(SCMC_sigmoid_kernel, (String,String)) # Time: 0.453531
    Base.precompile(SCMC_mult_kernel, (String,String,String)) # Time: 0.123286
    Base.precompile(SCMC_add_kernel, (String,String,String)) # Time: 0.008387

    # Main kernel generator
    Base.precompile(kgen, (Num, Vector{Num}, Vector{Symbol}, Vector{Num}, Bool)) # Time: 1.632566

    # Kernel generator requirements
    Base.precompile(create_kernel!, (String, Int, BasicSymbolic{Real}, Vector{Num}, Vector{Symbol}, Vector{Num})) # Time: 0.000162
    Base.precompile(perform_substitutions, (Vector{Equation},)) # Time: 0.000006
    Base.precompile(cleaner, (BasicSymbolic, Vector{Equation}, Vector{Bool})) # Time: 0.000007
    Base.precompile(preamble_string, (String, Vector{String}, Int, Int, Int)) # Time: 0.000005
    Base.precompile(write_operation, (IOStream, BasicSymbolic{Real}, Vector{Any}, Vector{String})) # Time: 0.000008
    Base.precompile(postamble, (String,)) # Time: 0.000004
    Base.precompile(outro, (String, Vector{Int}, Int32, Vector{Num})) # Time: 0.000002
end