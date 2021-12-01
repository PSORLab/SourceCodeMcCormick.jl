#=
Mostly garbage but saved for posterity

# Takes the right-hand-side function of the equation and computes an list of expressions which evaluated to lower bound via interval arithmetic
function substitute_interval_bnds(eq::Equation, x, xIntv, xv, j)
    d = Dict()
    for i = 1:length(x)
        if i == j
            d[x[i]] = xv
        else
            d[x[i]] = xIntv[i]
        end
    end
    eq = substitute(eq, d)
end

function substitute_interval_bnds(eq::Equation, x, xIntv)
    d = Dict()
    for i = 1:length(x)
        d[x[i]] = xIntv[i]
    end
    substitute(eq, d)
end

@parameters σ ρ β
@variables t x(t) y(t) z(t)
D = Differential(t)
eqs = [D(x) ~ σ*(y-x),
       D(y) ~ x*(ρ-z)-y,
       D(z) ~ x*y - β*z]
@named de = ODESystem(eqs,t,[x,y,z],[σ,ρ,β])

mutable struct ODERelaxProbMTK
    odes::ODESystem
    domains::Vector{Any}
end

rde = ODERelaxProbMTK(de, [])

function rewrite_modeling_toolkit_ode(prob::ODERelaxProbMTK)
    dx = equations(prob.odes)
    x = states(prob.odes)
    p = parameters(prob.odes)
    t = get_iv((prob.odes)

    nx = length(dx)

    @variables xlo[1:nx]
    @variables xhi[1:nx]
    @variables xIntv[1:nx]

    @parameters pli[1:np]
    @parameters pui[1:np]
    @parameters pIntvi[1:np]

    ode_rhs = quote
        tIntv = Interval(t)
    end

    # make P intervals
    for i = 1:np
        pli = Symbol("pl$i")
        pui = Symbol("pu$i")
        pIntvi = Symbol("P$i")
        ode_rhs = quote
            $ode_rhs
            $pli = p[$i]
            $pui = p[$np + $i]
            $pIntvi = Interval($pli, $pui)
        end
    end

    # make state intervals
    for i = 1:nx
        xli = Symbol("xl$i")
        xui = Symbol("xu$i")
        xIntvi = Symbol("xIntv$i")
        ode_rhs = quote
            $ode_rhs
            $xli = x[$i]
            $xui = x[$nx + $i]
            $xIntvi = Interval($xli, $xui)
        end
    end

    # subsititute interval variables into rhs as appropriate
    for (i, dxi) in enumerate(dx)
        dxip = substitute_interval_bnds(dxi, p, pIntv)
        dxi_l_eq = substitute_interval_bnds(dxip, x, xIntv, xlo[i], i)
        dxi_u_eq = substitute_interval_bnds(dxip, x, xIntv, xhi[i], i)
        ode_rhs = quote
            $ode_rhs
            dx[$i] = $(lo(rhs(dxi_l_eq)))
            dx[$i + $nx] = $(hi(rhs(dxi_u_eq)))
        end
    end
    return ode_rhs
end

ode_rhs = rewrite_modeling_toolkit_ode(prob::ODERelaxProbMTK)
@eval function f1(dx, x, p, t)
    $ode_rhs
    return
end

#=
using DifferentialEquations, IntervalArithmetic
f(u,p,t) = 1.01*u
u0 = 1/2
tspan = (0.0,1.0)
prob = ODEProblem(f,u0,tspan)
sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8)

function prob_func(prob,i,repeat)
    @. prob.u0 = randn()*prob.u0
    prob
  end

p = EnsembleProblem(prob; prob_func = prob_func)
sim = solve(p, Tsit5(), EnsembleThreads(), trajectories=1000)
=#
