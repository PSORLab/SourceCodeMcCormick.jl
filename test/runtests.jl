using SourceCodeMcCormick, McCormick, Symbolics, Test, IfElse

function eval_check(eval_func, MC1, MC2)
    return eval_func(MC1.cv, MC1.cc, MC1.Intv.lo, MC1.Intv.hi,
                     MC2.cv, MC2.cc, MC2.Intv.lo, MC2.Intv.hi)
end
function eval_check(eval_func, MC1)
    return eval_func(MC1.cv, MC1.cc, MC1.Intv.lo, MC1.Intv.hi)
end
include("multiplication.jl")
include("division.jl")
include("addition.jl")
include("exp.jl")
include("power.jl") #NOTE: Currently only includes ^2
# include("minmax.jl")