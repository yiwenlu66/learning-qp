## Read parameters

using NPZ

data = npzread("parameters.npz")
W_q = data["Wq"]
P = data["P"]
A_MCI = data["A_MCI"]
b_MCI = data["b_MCI"]
H = data["H"]
A = data["A"]
B = data["B"]
m_mci, n_sys = size(A_MCI)
n_qp = size(P, 1)
m_qp = size(H, 1)
m_sys = size(B, 2)

norm_factor = 0.5

## Define the candidate invariant set to be tested
relax = 0.2
G = A_MCI
c = b_MCI .- relax

## Formulate the problem of verifying whether a set is invariant into a bilevel optimization problem, and try solving using BilevelJuMP

using JuMP
using BilevelJuMP
using Ipopt

# Define the bilevel model
blmodel = BilevelModel(Ipopt.Optimizer; mode = BilevelJuMP.ProductMode(1e-9))

# Upper level
@variable(Upper(blmodel), x[1:n_sys], start = 1)
@variable(Upper(blmodel), λ[1:m_mci] >= 0)
@variable(Lower(blmodel), u[1:n_qp])
@constraint(Upper(blmodel), sum(λ) == 1)
@constraint(Upper(blmodel), G * x .<= c)
@objective(Upper(blmodel), Min, -λ' * (G * (A * x + norm_factor * B * u[1:m_sys]) - c))


# Lower level
@constraint(Lower(blmodel), H*u .<= 1.)
@constraint(Lower(blmodel), -1. .<= H*u)
@constraint(Lower(blmodel), u[1:m_sys] .<= 1.)
@constraint(Lower(blmodel), -1. .<= u[1:m_sys])
@objective(Lower(blmodel), Min, 0.5 * u' * P * u + x' * W_q' * u)

# Solve the bilevel problem
optimize!(blmodel)

# Extract results
optimal_value = objective_value(blmodel)   # Is it correct?
# @show optimal_value
@show value.(x)
@show value.(u)
@show value.(λ)
optimal_value = -value.(λ)' * (G * (A * value.(x) + norm_factor * B * value.(u)[1:m_sys]) - c)
@show optimal_value

## Visualize
using Polyhedra, CDDLib, Plots, Statistics

function sort_vertices(vertices)
    # Calculate centroid
    centroid = mean(vertices, dims=1)

    vertices = [vertices[i, :] for i in 1:size(vertices, 1)]

    # Sort vertices based on polar angle from centroid
    sorted_vertices = sort(vertices, by = p -> atan(p[2] - centroid[2], p[1] - centroid[1]))

    return sorted_vertices
end

function plot_polytope(A, b, fig, label)
    poly = polyhedron(hrep(A, b), CDDLib.Library())
    v = sort_vertices(hcat(points(vrep(poly))...)')
    x_coords = [x[1] for x in v]
    y_coords = [x[2] for x in v]

    Plots.scatter!(fig, x_coords, y_coords, label = label)

    for i = 1:length(v)
        Plots.plot!(fig, [x_coords[i], x_coords[(i % length(v)) + 1]], [y_coords[i], y_coords[(i % length(v)) + 1]], color="black", label="")
    end
    fig
end

fig = Plots.plot()
plot_polytope(A_MCI, b_MCI, fig, "MCI")
plot_polytope(G, c, fig, "Verified")
Plots.scatter!(fig, [value.(x)[1]], [value.(x)[2]], label = "Worst case", color = "green")

## Try SDP Lower bound with Lagrangian relaxation
using JuMP, SCS
using LinearAlgebra

my = 2 * m_mci + 2   # Number of constraints for outer problem
mx = 2 * m_qp + 2 * m_sys  # Number of constraints for inner problem
ny = n_sys + m_mci  # Number of variables for outer problem
nx = n_qp  # Number of variables for inner problem

E = [Matrix(1.0I, m_sys, m_sys) zeros(m_sys, n_qp - m_sys)]  # Matrix for extracing u from QP solution
Rxx = zeros(n_qp, n_qp)
Rxy = [zeros(n_qp, n_sys) E' * B' * G']
Ryx = Rxy'
Ryy = [zeros(n_sys, n_sys)  -A' * G'; -G * A  zeros(m_mci, m_mci)]
sx = zeros(n_qp)
sy = [zeros(n_sys); c]
Gx = zeros(my, nx)
Gy = [-G zeros(m_mci, m_mci);
      zeros(m_mci, n_sys) Matrix(1.0I, m_mci, m_mci);
      zeros(1, n_sys) ones(1, m_mci)
      zeros(1, n_sys) -ones(1, m_mci)
    ]
c̃ = [c; zeros(m_mci); -1; 1]
bq = zeros(n_qp)
H̃ = [H; -H; E; -E]
b̃ = ones(mx)
Wb = zeros(mx, ny)
Wq = [W_q zeros(n_qp, m_mci)]


# Initialize the model with SCS solver
model = Model(optimizer_with_attributes(SCS.Optimizer))

# Define variables
@variable(model, γ)
@variable(model, μ1[1:my] >= 0)
@variable(model, μ2[1:mx] >= 0)
@variable(model, μ3[1:mx] >= 0)
@variable(model, η1[1:nx])
@variable(model, η2)
@variable(model, η3)
@variable(model, η4)

# Objective function
@objective(model, Max, γ)

# SDP constraint
function make_M(γ, μ1, μ2, μ3, η1, η2, η3, η4)
    up = [
        Rxx + 2*η4*P    Rxy + η4*Wq    η2*H̃'  sx - Gx'*μ1 - H̃'*μ2 + P*η1 + η4*bq;
        zeros(ny, nx)      Ryy       η2*Wb' + η3*(Wb' - Wq'*inv(P)*H̃') + η4*Wb'   sy - Gy'*μ1 - Wb'*μ2 + Wq'*η1;
        zeros(mx, nx)      zeros(mx, ny)          2*η3*H̃*inv(P)*H̃'  -μ3 - H̃*η1 + η2*b̃ + η3*(b̃ - H̃*inv(P)*bq) + η4*b̃;
        zeros(1, nx)      zeros(1, ny)          zeros(1, mx)             -2*c̃'*μ1 - 2*b̃'*μ2 + 2*bq'*η1 - γ
    ]
    return (up + up') / 2
end

@expression(model, M, make_M(γ, μ1, μ2, μ3, η1, η2, η3, η4))
@constraint(model, M in PSDCone())

# Solve the problem
optimize!(model)

# It will be infeasible

## Formulate nonconvex QCQP and try solving using local solver

using JuMP, Ipopt

model = Model(Ipopt.Optimizer)

@variable(model, x[1:n_sys], start = 1)
@variable(model, λ[1:m_mci] >= 0)
@variable(model, u[1:n_qp])
@variable(model, μ1[1:m_qp] >= 0)
@variable(model, μ2[1:m_qp] >= 0)
@variable(model, μ3[1:m_sys] >= 0)
@variable(model, μ4[1:m_sys] >= 0)

p = -λ' * (G * (A * x + norm_factor * B * u[1:m_sys]) - c)
@NLobjective(model, Min, p)
@constraint(model, G * x .<= c)
@constraint(model, sum(λ) == 1)
@constraint(model, P * u + W_q * x + H' * (μ1 - μ2) .== 0)
@constraint(model, H * u .<= 1)
@constraint(model, -1 .<= H * u)
@constraint(model, u[1:m_sys] .<= 1)
@constraint(model, -1 .<= u[1:m_sys])
@constraint(model, μ1' * (H * u .- 1) == 0)
@constraint(model, μ2' * (-H * u .- 1) == 0)
@constraint(model, μ3' * (u[1:m_sys] .- 1) == 0)
@constraint(model, μ4' * (-u[1:m_sys] .- 1) == 0)
optimize!(model)
# @show objective_value(model)
@show value.(x)
@show value.(u)
@show value.(λ)
optimal_value = -value.(λ)' * (G * (A * value.(x) + norm_factor * B * value.(u)[1:m_sys]) - c)
@show optimal_value

## Try lower bound with SOS solver

using DynamicPolynomials, SumOfSquares
# import SCS
# scs = SCS.Optimizer
import MosekTools
mosek = MosekTools.Optimizer
import Dualization
# dual_scs = Dualization.dual_optimizer(scs)
# model = SOSModel(dual_scs)
dual_mosek = Dualization.dual_optimizer(mosek)
model = SOSModel(dual_mosek)

@polyvar x[1:n_sys]
@polyvar λ[1:m_mci]
@polyvar u[1:n_qp]
@polyvar μ1[1:m_qp]
@polyvar μ2[1:m_qp]
@polyvar μ3[1:m_sys]
@polyvar μ4[1:m_sys]

p = -λ' * (G * (A * x + norm_factor * B * u[1:m_sys]) - c)
S = BasicSemialgebraicSet{Float64,Polynomial{true,Float64}}()

invariance_constraint = - (G * x - c)
for i in 1:m_mci
    addinequality!(S, invariance_constraint[i])
end
for i in 1:m_mci
    addinequality!(S, λ[i])
end
addequality!(S, sum(λ) - 1)
stationarity = P * u + W_q * x + H' * (μ1 - μ2)
for i in 1:n_qp
    addequality!(S, stationarity[i])
end
p_feasibility_1 = -(H * u .- 1)
for i in 1:m_qp
    addinequality!(S, p_feasibility_1[i])
end
p_feasibility_2 = H * u .+ 1
for i in 1:m_qp
    addinequality!(S, p_feasibility_2[i])
end
p_feasibility_3 = 1. .- u[1:m_sys]
for i in 1:m_sys
    addinequality!(S, p_feasibility_3[i])
end
p_feasibility_4 = u[1:m_sys] .+ 1.
for i in 1:m_sys
    addinequality!(S, p_feasibility_4[i])
end
for i in 1:m_qp
    addinequality!(S, μ1[i])
end
for i in 1:m_qp
    addinequality!(S, μ2[i])
end
for i in 1:m_sys
    addinequality!(S, μ3[i])
end
for i in 1:m_sys
    addinequality!(S, μ4[i])
end
addequality!(S, μ1' * (H * u .- 1))
addequality!(S, μ2' * (-H * u .- 1))
addequality!(S, μ3' * (u[1:m_sys] .- 1))
addequality!(S, μ4' * (-u[1:m_sys] .- 1))

@variable(model, σ >= 0)
@objective(model, Max, σ)
@constraint(model, p >= σ, domain = S, maxdegree = 3)
optimize!(model)
@show solution_summary(model)
@show objective_value(model)

## Some toy examples that exemplify the solver usage

##
using DynamicPolynomials, SumOfSquares
import MosekTools
mosek = MosekTools.Optimizer
import Dualization
dual_mosek = Dualization.dual_optimizer(mosek)

# Create JuMP model
model = SOSModel(dual_mosek)

@polyvar x y
p = x * y
@variable(model, σ)
@objective(model, Max, σ)
S = @set x + y <= 1 && x - y <= 1 && -x + y <= 1 && -x - y <= 1
# @constraint(model, x + y <= 1)
# @constraint(model, x - y <= 1)
# @constraint(model, -x + y <= 1)
# @constraint(model, -x - y <= 1)
@constraint(model, p >= σ, domain = S, maxdegree = 3)
optimize!(model)
solution_summary(model)

##
using DynamicPolynomials
@polyvar x y
p = x^3 - x^2 + 2x*y -y^2 + y^3
using SumOfSquares
S = @set x >= 0 && y >= 0 && x + y >= 1
import Ipopt
model = Model(Ipopt.Optimizer)
@variable(model, a >= 0)
@variable(model, b >= 0)
@constraint(model, a + b >= 1)
@NLobjective(model, Min, a^3 - a^2 + 2a*b - b^2 + b^3)
optimize!(model)
solution_summary(model)

##
import MosekTools
mosek = MosekTools.Optimizer
import Dualization
dual_mosek = Dualization.dual_optimizer(mosek)
model = SOSModel(dual_mosek)
@variable(model, α)
@objective(model, Max, α)
@constraint(model, c3, p >= α, domain = S)
optimize!(model)
solution_summary(model)

##
model = SOSModel(dual_mosek)
@variable(model, α)
@objective(model, Max, α)
@constraint(model, c4, p >= α, domain = S, maxdegree = 4)
optimize!(model)
solution_summary(model)

