## Read parameters

using NPZ

data = npzread("parameters.npz")
W_q = data["Wq"]
P = data["P"]
G = data["A_MCI"]
c = data["b_MCI"] .- 0.5
H = data["H"]
A = data["A"]
B = data["B"]
m_mci, n_sys = size(G)
n_qp = size(P, 1)
m_sys = size(B, 2)

norm_factor = 0.5

## Formulate the problem of verifying whether a set is invariant into a bilevel optimization problem, and try solving using BilevelJuMP

using JuMP
using BilevelJuMP
using Ipopt

# Define the bilevel model
blmodel = BilevelModel(Ipopt.Optimizer; mode = BilevelJuMP.ProductMode(1e-9))

# Upper level
@variable(Upper(blmodel), x[1:n_sys])
@variable(Upper(blmodel), λ[1:m_mci] >= 0)
@variable(Lower(blmodel), -1 <= u[1:n_qp] <= 1)
@constraint(Upper(blmodel), sum(λ) == 1)
@constraint(Upper(blmodel), G * x .<= c)
@objective(Upper(blmodel), Min, -λ' * (G * (A * x + norm_factor * B * u[1:m_sys]) - c))


# Lower level
@constraint(Lower(blmodel), H*u .<= 1.)
@constraint(Lower(blmodel), -1. .<= H*u)
@objective(Lower(blmodel), Min, 0.5 * u' * P * u + x' * W_q' * u)

##
# Solve the bilevel problem
optimize!(blmodel)

# Extract results
optimal_value = objective_value(blmodel)
@show optimal_value

##
value.(x)
