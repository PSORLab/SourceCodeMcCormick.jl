using SourceCodeMcCormick, McCormick, Symbolics, Test, IfElse

function eval_check(eval_func, MC1, MC2)
    return eval_func(MC1.cv, MC1.cc, MC1.Intv.lo, MC1.Intv.hi,
                     MC2.cv, MC2.cc, MC2.Intv.lo, MC2.Intv.hi)
end
function eval_check_grad(eval_func, MC1, MC2)
    return eval_func(MC1.cv, MC1.cc, MC1.Intv.lo, MC1.Intv.hi, MC1.cv_grad[1], MC1.cv_grad[2], MC1.cc_grad[1], MC1.cc_grad[2],
                     MC2.cv, MC2.cc, MC2.Intv.lo, MC2.Intv.hi, MC2.cv_grad[1], MC2.cv_grad[2], MC2.cc_grad[1], MC2.cc_grad[2])
end
function eval_check(eval_func, MC1)
    return eval_func(MC1.cv, MC1.cc, MC1.Intv.lo, MC1.Intv.hi)
end
function eval_check_grad(eval_func, MC1)
    return eval_func(MC1.cv, MC1.cc, MC1.Intv.lo, MC1.Intv.hi, MC1.cv_grad[1], MC1.cc_grad[1])
end
include("multiplication.jl")
include("division.jl")
include("addition.jl")
include("exp.jl")
include("power.jl") #NOTE: Currently only includes ^2
# include("minmax.jl")