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
m_mci, n_sys = size(G)
n_qp = size(P, 1)
m_qp = size(H, 1)
m_sys = size(B, 2)

norm_factor = 0.5

## Define the candidate invariant set to be tested
relax = 1
G = A_MCI
c = b_MCI .- relax

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
@show value.(x)

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

##
fig = Plots.plot()
plot_polytope(A_MCI, b_MCI, fig, "MCI")
plot_polytope(G, c, fig, "Verified")
Plots.scatter!(fig, [value.(x)[1]], [value.(x)[2]], label = "Worst case", color = "green")

## Try SDP Lower bound
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


