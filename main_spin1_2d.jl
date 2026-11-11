##########################################################################################################
# A Riemannian Optimization Framework for Spinor Bose-Einstein Condensates
# P. Henning, L. Huynh, T. Stykel, M. Yadav 
# 2D FEM discretization (P1 and P2) of Riemannian descent methods to compute ground states of SPIN-1 BECs
##########################################################################################################

# The code can be run on multiple threads. Start julia with e.g. "julia -t 8"
# Main program can be found at the very bottom of this code.


using LinearAlgebra
using SparseArrays
using StaticArrays
using Printf
using CairoMakie
using Base.Threads

struct Params{FType}
    order_finite_element::Int
    a::Float64
    b::Float64
    F::Int # spin
    N::Int 
    β0::Float64
    β1::Float64
    p::Float64 
    q::Float64
    V_fun::FType
    C_mag::Float64 
    maxIter::Int
    tol::Float64  
end

struct Mesh   
    T::Matrix{Int}
    x::Matrix{Float64}
    bn::Vector{Int}
    global_to_interior::Vector{Int}
end

struct SpinorState  
    u1::Vector{ComplexF64}
    u0::Vector{ComplexF64}
    um1::Vector{ComplexF64}
end

struct SpinMatrices{F, Dim, M <: SMatrix{Dim, Dim, ComplexF64}}
    Sx::M 
    Sy::M 
    Sz::M
end

function SpinMatrices(F::Int)
    Dim = 2 * F + 1

    if F == 1
        inv_sqrt2 = 1.0 / sqrt(2.0)
        
        Sx_s = @SMatrix ComplexF64[
            0.0        inv_sqrt2  0.0;
            inv_sqrt2  0.0        inv_sqrt2;
            0.0        inv_sqrt2  0.0
        ]

        Sy_s = @SMatrix ComplexF64[
            0.0         -im*inv_sqrt2  0.0;
            im*inv_sqrt2 0.0          -im*inv_sqrt2;
            0.0          im*inv_sqrt2  0.0
        ]

        Sz_s = @SMatrix ComplexF64[
            1.0  0.0   0.0;
            0.0  0.0   0.0;
            0.0  0.0  -1.0
        ]
    else
        error("Unsupported spin value F = $F")
    end

    M = typeof(Sx_s)
    return SpinMatrices{F, Dim, M}(Sx_s, Sy_s, Sz_s)
end

function to_vec!(out::Vector{ComplexF64}, u::SpinorState)
    n = length(u.u1)
    @inbounds begin
        out[1:n]        .= u.u1
        out[n+1:2n]     .= u.u0
        out[2n+1:3n]    .= u.um1
    end
    return out
end

function from_vec!(u_out::SpinorState, v::AbstractVector)
    n = length(v) ÷ 3
    @inbounds begin
        u_out.u1  .= @view(v[1:n])
        u_out.u0  .= @view(v[n+1:2n])
        u_out.um1 .= @view(v[2n+1:3n])
    end
    return u_out
end

function Base.:+(u::SpinorState, v::SpinorState)
    sum_spinor_state = SpinorState(u.u1 + v.u1, u.u0 + v.u0, u.um1 + v.um1)
    return sum_spinor_state
end

function P2_Tri2Edge(Nd_P1, T_P1)
    np = size(Nd_P1, 1)
    nt = size(T_P1, 1)
    i = T_P1[:, 1] #first vertex
    j = T_P1[:, 2]
    k = T_P1[:, 3]

    #= adjacency matrix: entries represent connections between nodes
    the three edges of each triangle are stored as -1 (i,e. A(j,k) = A(i,k) = A(i,j) = -1) =#
    
    A = sparse(j, k, fill(-1, length(j)), np, np) #A(j(index),k(index)) = -1, i.e. there is an edge between the nodes j(index) and k(index)
    A = A + sparse(i, k, fill(-1, length(i)), np, np) #2nd edge
    A = A + sparse(i, j, fill(-1, length(i)), np, np) #3rd edge

   # force A to be symmetric, keep only upper triangle where condition is met
    A = -((A + A') .< 0)       
    A = triu(A)                 

    # Assign unique edge indices
    r, c, _ = findnz(A)         # get row & col indices of nonzero entries
    v = collect(1:length(r))    # unique edge indices
    A = sparse(r, c, v, np, np) # create sparse matrix with edge indices
    A = A + A'                  # make symmetric again, so A(i,j) = A(j,i) = edge index

    # edge indices for each triangle
    edges = zeros(Int, nt, 3)   

    for l in 1:nt
        i, j, k = T_P1[l, 1], T_P1[l, 2], T_P1[l, 3]
        edges[l, :] = [A[i, j], A[j, k], A[k, i]]
    end
    return edges
end

function unique_indices(v)
    seen = Dict{eltype(v), Int}()
    u = eltype(v)[]
    idx = Int[]

    for (i, val) in pairs(v)
        if !haskey(seen, val)
            seen[val] = i
            push!(u, val)
            push!(idx, i)
        end
    end

    order = sortperm(u)
    return u[order], idx[order]
end

function P2_EdgeMidPoints(Nd_P1::Matrix{Float64}, edges::Matrix{Int}, T_P1::Matrix{Int})
    edges1 = edges[:]

    i = T_P1[:, 1]
    j = T_P1[:, 2]
    k = T_P1[:, 3]

    start = [i j k]
    stop  = [j k i]

    xmid_all = (Nd_P1[start, 1] .+ Nd_P1[stop, 1]) ./ 2
    ymid_all = (Nd_P1[start, 2] .+ Nd_P1[stop, 2]) ./ 2
    # remove duplicate edges 
    _, idx = unique_indices(edges1)

    # extract only unique midpoints
    xmid = xmid_all[idx]
    ymid = ymid_all[idx]

    return xmid, ymid
end

function fem2d_uniform_mesh(params::Params)
    order_finite_element = params.order_finite_element
    N = params.N 
    a = params.a 
    b = params.b

    h = (b - a) / N  

    num_nodes = (N + 1)^2
    num_tris = 2 * N^2

    T_P1 = Matrix{Int}(undef, num_tris, 3)
    Nd_P1 = Matrix{Float64}(undef, num_nodes, 2)
    bn_P1 = zeros(Int, num_nodes)

    θ = 15
    R = [cosd(θ) -sind(θ); sind(θ) cosd(θ)]

    for j in 0:N, i in 0:N
        idx = i + j * (N + 1) + 1
        Nd_P1[idx, :] = [a + i * h, a + j * h]
        bn_P1[idx] = (i == 0 || i == N || j == 0 || j == N) ? 1 : 0
    end

    counter_T = 1
    for j in 0:N-1, i in 1:N
        idx = i + j * (N + 1)
        T_P1[counter_T, :] = [idx, idx + N + 2, idx + N + 1]
        counter_T += 1
        T_P1[counter_T, :] = [idx, idx + 1, idx + N + 2]
        counter_T += 1
    end

    if order_finite_element == 2 # P2 finite elements
            T_P2 = zeros(Int64, size(T_P1,1),6)
            T_P2[:, [1 3 5]] = T_P1

            Nd_P2 = Nd_P1
            Nx = size(Nd_P1, 1)

            edges = P2_Tri2Edge(Nd_P1, T_P1)

            xmid, ymid = P2_EdgeMidPoints(Nd_P1, edges, T_P1)
            # assign new indices, starting from Nx + 1
            edges_new = edges + Nx*ones(size(edges)) 
            T_P2[:, [2 4 6]] = edges_new
            # append coordinates of midpoints
            Nd_P2 = [Nd_P2;[xmid ymid]] 
            bn_P2 = zeros(Int,size(Nd_P2,1))
            bn_P2[1:length(bn_P1)] = bn_P1

            #check if boundary node
            for l = length(bn_P1)+1 : size(Nd_P2,1)
                x = Nd_P2[l,1]
                y = Nd_P2[l,2]
                if abs(x) == abs(a) || abs(x) == abs(b) || abs(y) == abs(a) || abs(y) == abs(b)
                    bn_P2[l] = 1
                end
            end

        end

    if order_finite_element == 1
            T = T_P1 
            x = Nd_P1 
            bn = bn_P1
        elseif order_finite_element == 2
            T = T_P2 
            x = Nd_P2
            bn = bn_P2
    end

        interior = findall(bn .== 0)
         global_to_interior = zeros(Int, size(bn, 1))
         for (i, idx) in enumerate(interior)
             global_to_interior[idx] = i
         end


    return Mesh(T, x, bn, global_to_interior)
end

function matrices(mesh, params) # mass, stiffness and potential matrices
    V_fun = params.V_fun 
    T = mesh.T 
    bn = mesh.bn 
    Nd = mesh.x
    inner_indices = mesh.global_to_interior

    order_finite_element = params.order_finite_element

    if order_finite_element == 1
        phi = [
        x -> -x[1] - x[2] + 1
        x -> x[1]
        x -> x[2]
        ]

        grad_phi = [
        x -> [-1; -1]
        x -> [1; 0]
        x -> [0; 1]
        ]
        num_entries = 9*size(T, 1)

    elseif order_finite_element == 2
        phi = [
        x -> (1 - x[1] - x[2]) * (1 - 2 * x[1] - 2 * x[2]),
        x -> 4 * x[1] * (1 - x[1] - x[2]),
        x -> x[1] * (2 * x[1] - 1),
        x -> 4 * x[1] * x[2],
        x -> x[2] * (2 * x[2] - 1),
        x -> 4 * x[2] * (1 - x[1] - x[2]),
        ]

        grad_phi = [
        (x -> [4 * x[1] + 4 * x[2] - 3, 4 * x[1] + 4 * x[2] - 3]),
        (x -> [-4 * (2 * x[1] + x[2] - 1), -4 * x[1]]),
        (x -> [4 * x[1] - 1, 0]),
        (x -> [4 * x[2], 4 * x[1]]),
        (x -> [0, 4 * x[2] - 1]),
        (x -> [-4 * x[2], -4 * (x[1] + 2 * x[2] - 1)])
        ]
        num_entries = 36*size(T, 1)
    end

    m_i = zeros(Int, num_entries)
    m_j = zeros(Int, num_entries)

    m = zeros(Float64, num_entries)  
    s = zeros(Float64, num_entries)   
    v = zeros(Float64, num_entries)

    w = @SVector [0.225, 0.132394152788506, 0.132394152788506, 0.132394152788506, 0.125939180544827, 0.125939180544827, 0.125939180544827]
    α = @SVector [0.059715871789770, 0.797426985353087]
    β = @SVector [0.470142064105115, 0.101286507323456]

    x_quad = (
        SVector(1/3, 1/3),
        SVector(β[1], β[1]), 
        SVector(α[1], β[1]), 
        SVector(β[1], α[1]),
        SVector(β[2], β[2]), 
        SVector(α[2], β[2]), 
        SVector(β[2], α[2])
    )

    no_of_basis = size(phi, 1)

    phi_in_quad_point = zeros(no_of_basis, 7)
    grad_phi_in_quad_point = zeros(2, no_of_basis, 7)

    for i = 1:no_of_basis
        for q = 1:7
            phi_in_quad_point[i, q] = phi[i](x_quad[q])
            grad_phi_in_quad_point[:, i, q] = grad_phi[i](x_quad[q])
        end
    end
     ind = 1
    for k in axes(T, 1)
        tri = T[k, :]
        z1 = SVector(Nd[tri[1], 1], Nd[tri[1], 2])
        
        if order_finite_element == 1
            z2 = SVector(Nd[tri[2], 1], Nd[tri[2], 2])
            z3 = SVector(Nd[tri[3], 1], Nd[tri[3], 2])
        elseif order_finite_element == 2
            z2 = SVector(Nd[tri[3], 1], Nd[tri[3], 2]) 
            z3 = SVector(Nd[tri[5], 1], Nd[tri[5], 2]) 
        end

        z = [z1[1]; z1[2]]

        BT = [z2[1]-z1[1] z3[1]-z1[1]; 
            z2[2]-z1[2] z3[2]-z1[2]
        ]
        detBT = 0.5 * (BT[1, 1] * BT[2, 2] - BT[1, 2] * BT[2, 1])
        BTinv = transpose(inv(BT))

        for i = 1:no_of_basis
            for j = 1:no_of_basis
                if (i <= j)
                    mass = 0.0
                    stiffness = 0.0
                    potential = 0.0
                    for l = 1:7
                        mass += w[l] * phi_in_quad_point[i, l] * phi_in_quad_point[j, l]
                        stiffness += w[l] * dot(BTinv * grad_phi_in_quad_point[:, i, l],  BTinv*grad_phi_in_quad_point[:, j, l])
                        potential += w[l]*V_fun(BT*x_quad[l] + z) * phi_in_quad_point[i,l]*phi_in_quad_point[j,l]
                    end
                    mass *= detBT 
                    stiffness *= detBT
                    potential *= detBT

                    if bn[tri[i]] == 0 && bn[tri[j]] == 0
   
                        m_i[ind] = inner_indices[tri[i]]
                        m_j[ind] = inner_indices[tri[j]]

                        m[ind] = mass
                        s[ind] = stiffness
                        v[ind] = potential

                        ind += 1
                        
                        if i < j #symmetric structutre
                            m_i[ind] = inner_indices[tri[j]]
                            m_j[ind] = inner_indices[tri[i]]

                            m[ind] = mass
                            s[ind] = stiffness
                            v[ind] = potential

                            ind += 1
                        end
                    end
                end
            end
        end
    end

        del_index = findfirst(x -> x == 0, m_i)
        M = sparse(m_i[1:del_index - 1], m_j[1:del_index - 1], m[1:del_index - 1])
        S = sparse(m_i[1:del_index - 1], m_j[1:del_index - 1], s[1:del_index - 1])
        V = sparse(m_i[1:del_index - 1], m_j[1:del_index - 1], v[1:del_index - 1])

    return M, S, V
    
end

function initial_phi(mesh, ζ, M)
    bn = mesh.bn 
    x = mesh.x 
    interior = findall(bn .== 0) #find all interior nodes

    # extract interior nodes
    x_inner = x[interior, :]

    phi = [ζ(x_inner[i,:]) for i in axes(x_inner,1)]
    
    phi_norm = sqrt(real(phi' * M * phi))
    phi = phi / phi_norm

    return phi
    
end

function spin1_energy(u, mesh, params, spin_mats)
    u1 = u.u1
    u0 = u.u0
    um1 = u.um1

    Sx = spin_mats.Sx
    Sy = spin_mats.Sy
    Sz = spin_mats.Sz

    V_fun = params.V_fun
    order_finite_element = params.order_finite_element
    x_mesh = mesh.x
    T = mesh.T
    bn = mesh.bn
    global_to_interior = mesh.global_to_interior

    no_of_basis = (order_finite_element == 1) ? 3 : 6

    β0 = params.β0
    β1 = params.β1
    q_param = params.q
    p_param = params.p

    w = @SVector [0.225, 0.132394152788506, 0.132394152788506, 0.132394152788506, 0.125939180544827, 0.125939180544827, 0.125939180544827]
    α = @SVector [0.059715871789770, 0.797426985353087]
    β = @SVector [0.470142064105115, 0.101286507323456]
    
    x_quad = (
        SVector(1 / 3, 1 / 3),
        SVector(β[1], β[1]),
        SVector(α[1], β[1]),
        SVector(β[1], α[1]),
        SVector(β[2], β[2]),
        SVector(α[2], β[2]),
        SVector(β[2], α[2]),
    )
    no_of_quad_points = length(x_quad)

    E_kin = 0.0
    E_pot = 0.0
    E_zeeman = 0.0
    E_rho = 0.0
    E_spin = 0.0

    @inbounds for k in axes(T, 1)
        tri1 = T[k, 1]
        tri3 = (order_finite_element == 1) ? T[k, 2] : T[k, 3]
        tri5 = (order_finite_element == 1) ? T[k, 3] : T[k, 5]

        z1 = SVector(x_mesh[tri1, 1], x_mesh[tri1, 2])
        z2 = SVector(x_mesh[tri3, 1], x_mesh[tri3, 2])
        z3 = SVector(x_mesh[tri5, 1], x_mesh[tri5, 2])

        J11 = z2[1] - z1[1]; J12 = z3[1] - z1[1]
        J21 = z2[2] - z1[2]; J22 = z3[2] - z1[2]

        detBT = 0.5 * (J11 * J22 - J12 * J21)
        invDet = 1.0 / (J11 * J22 - J12 * J21)

        iJ11 =  J22 * invDet; iJ12 = -J21 * invDet
        iJ21 = -J12 * invDet; iJ22 =  J11 * invDet

        u1_loc  = MVector{6, ComplexF64}(undef)
        u0_loc  = MVector{6, ComplexF64}(undef)
        um1_loc = MVector{6, ComplexF64}(undef)

        for i in 1:no_of_basis
            node_i = T[k, i]
            if bn[node_i] == 0
                gi = global_to_interior[node_i]
                u1_loc[i]  = ComplexF64(u1[gi])
                u0_loc[i]  = ComplexF64(u0[gi])
                um1_loc[i] = ComplexF64(um1[gi])
            else
                u1_loc[i]  = 0.0 + 0.0im
                u0_loc[i]  = 0.0 + 0.0im
                um1_loc[i] = 0.0 + 0.0im
            end
        end

        phi_quad = NTuple{7, SVector{6, Float64}}(
        begin
            xi1, xi2 = x_quad[qpt]

            if order_finite_element == 1
                SVector(
                    1.0 - xi1 - xi2,
                    xi1,
                    xi2,
                    0.0, 0.0, 0.0
                )
            else
                SVector(
                    (1.0 - xi1 - xi2) * (1.0 - 2.0 * xi1 - 2.0 * xi2),
                    4.0 * xi1 * (1.0 - xi1 - xi2),
                    xi1 * (2.0 * xi1 - 1.0),
                    4.0 * xi1 * xi2,
                    xi2 * (2.0 * xi2 - 1.0),
                    4.0 * xi2 * (1.0 - xi1 - xi2)
                )
            end
        end for qpt in 1:7
          )

           grad_phi_quad = NTuple{7, SVector{6, SVector{2, Float64}}}(
            begin
                xi1, xi2 = x_quad[qpt]

                if order_finite_element == 1
                    z = SVector(0.0, 0.0)

                    SVector(
                        SVector(-1.0, -1.0),
                        SVector( 1.0,  0.0),
                        SVector( 0.0,  1.0),
                        z, z, z
                    )
                else
                    SVector(
                        SVector(
                            4.0 * xi1 + 4.0 * xi2 - 3.0,
                            4.0 * xi1 + 4.0 * xi2 - 3.0
                        ),
                        SVector(
                            -4.0 * (2.0 * xi1 + xi2 - 1.0),
                            -4.0 * xi1
                        ),
                        SVector(4.0 * xi1 - 1.0, 0.0),
                        SVector(4.0 * xi2, 4.0 * xi1),
                        SVector(0.0, 4.0 * xi2 - 1.0),
                        SVector(
                            -4.0 * xi2,
                            -4.0 * (xi1 + 2.0 * xi2 - 1.0)
                        )
                    )
                        end
                    end for qpt in 1:7
                )

        for qpt in 1:no_of_quad_points
            xi = x_quad[qpt]
            wq = w[qpt]
            phi = phi_quad[qpt]
            grad_phi = grad_phi_quad[qpt]

            x_phy = J11 * xi[1] + J12 * xi[2] + z1[1]
            y_phy = J21 * xi[1] + J22 * xi[2] + z1[2]
            V_val = V_fun(SVector(x_phy, y_phy))

            uq1  = 0.0 + 0.0im
            uq0  = 0.0 + 0.0im
            uqm1 = 0.0 + 0.0im

            g_u1  = SVector(0.0 + 0.0im, 0.0 + 0.0im)
            g_u0  = SVector(0.0 + 0.0im, 0.0 + 0.0im)
            g_um1 = SVector(0.0 + 0.0im, 0.0 + 0.0im)

            for i in 1:no_of_basis
                phi_i = phi[i]
                uq1  += u1_loc[i]  * phi_i
                uq0  += u0_loc[i]  * phi_i
                uqm1 += um1_loc[i] * phi_i

                g_phys = SVector(
                    iJ11 * grad_phi[i][1] + iJ12 * grad_phi[i][2],
                    iJ21 * grad_phi[i][1] + iJ22 * grad_phi[i][2]
                )

                g_u1  += u1_loc[i]  * g_phys
                g_u0  += u0_loc[i]  * g_phys
                g_um1 += um1_loc[i] * g_phys
            end

            u_vec = SVector{3, ComplexF64}(uq1, uq0, uqm1)

            grad_sq = real(dot(g_u1, g_u1) + dot(g_u0, g_u0) + dot(g_um1, g_um1))

            rho = real(dot(u_vec, u_vec))

            Fx = real(dot(u_vec, Sx * u_vec))
            Fy = real(dot(u_vec, Sy * u_vec))
            Fz = real(dot(u_vec, Sz * u_vec))
            F_sq = Fx^2 + Fy^2 + Fz^2

            fac = wq * detBT

            E_kin    += fac * grad_sq
            E_pot    += fac * V_val * rho
            E_zeeman += fac * ((q_param - p_param) * abs2(uq1) + (q_param + p_param) * abs2(uqm1))
            E_rho    += fac * (rho^2)
            E_spin   += fac * F_sq
        end
    end

    E = 0.5 * E_kin + E_pot + E_zeeman + 0.5 * β0 * E_rho + 0.5 * β1 * E_spin
    return E
end

#workspace for buffers
struct Workspace
    g_out::SpinorState
    chunk_starts::Vector{Int}
    chunk_ends::Vector{Int}
    out_buffers::Vector{Vector{ComplexF64}}
    Mu1::Vector{ComplexF64}
    Mu0::Vector{ComplexF64}
    Mum1::Vector{ComplexF64}
    res_u1::Vector{ComplexF64}   
    res_u0::Vector{ComplexF64}   
    res_um1::Vector{ComplexF64}
    u1_evals::Vector{Vector{ComplexF64}}
    u0_evals::Vector{Vector{ComplexF64}}
    um1_evals::Vector{Vector{ComplexF64}}
    
    cg_b::Vector{ComplexF64}
    cg_x::Vector{ComplexF64}
    cg_r::Vector{ComplexF64}
    cg_r_new::Vector{ComplexF64}
    cg_p::Vector{ComplexF64}
    cg_Ap_out::Vector{ComplexF64}
    cg_z::Vector{ComplexF64}

    u_vec::Vector{ComplexF64}
    out::Vector{ComplexF64}
    sobolev_grad::SpinorState
    ∇_A_N::SpinorState
    ∇_A_M::SpinorState

    u_proj::SpinorState
    MLu::SpinorState
end

function Workspace(u::SpinorState, n_elements::Int, order_finite_element::Int=1)
    n_comp = length(u.u1)
    n_threads = nthreads()
    n_flat = 3 * n_comp
    no_of_basis = (order_finite_element == 1) ? 3 : 6

    g_out = SpinorState(zeros(ComplexF64, n_comp), zeros(ComplexF64, n_comp), zeros(ComplexF64, n_comp))
    Mu1  = zeros(ComplexF64, n_comp)
    Mu0  = zeros(ComplexF64, n_comp)
    Mum1 = zeros(ComplexF64, n_comp)
    
    res_u1  = zeros(ComplexF64, n_comp)
    res_u0  = zeros(ComplexF64, n_comp)
    res_um1 = zeros(ComplexF64, n_comp)
    
    u1_evals  = [Vector{ComplexF64}(undef, no_of_basis) for _ in 1:n_threads]
    u0_evals  = [Vector{ComplexF64}(undef, no_of_basis) for _ in 1:n_threads]
    um1_evals = [Vector{ComplexF64}(undef, no_of_basis) for _ in 1:n_threads]
    
    out_buffers = [zeros(ComplexF64, n_flat) for _ in 1:n_threads]
    cg_b      = zeros(ComplexF64, n_flat)
    cg_x      = zeros(ComplexF64, n_flat)
    cg_r      = zeros(ComplexF64, n_flat)
    cg_r_new  = zeros(ComplexF64, n_flat)
    cg_p      = zeros(ComplexF64, n_flat)
    cg_Ap_out = zeros(ComplexF64, n_flat)
    cg_z      = zeros(ComplexF64, n_flat)

    u_vec = zeros(ComplexF64, n_flat)
    out   = zeros(ComplexF64, n_flat)
    sobolev_grad   = SpinorState(zeros(ComplexF64, n_comp), zeros(ComplexF64, n_comp), zeros(ComplexF64, n_comp))
    ∇_A_N = SpinorState(zeros(ComplexF64, n_comp), zeros(ComplexF64, n_comp), zeros(ComplexF64, n_comp))
    ∇_A_M = SpinorState(zeros(ComplexF64, n_comp), zeros(ComplexF64, n_comp), zeros(ComplexF64, n_comp))
    u_proj =  SpinorState(zeros(ComplexF64, n_comp), zeros(ComplexF64, n_comp), zeros(ComplexF64, n_comp))

    chunk_size = ceil(Int, n_elements / n_threads)
    c_starts = [(i - 1) * chunk_size + 1 for i in 1:n_threads]
    c_ends   = [min(i * chunk_size, n_elements) for i in 1:n_threads]

    active = [i for i in 1:n_threads if c_starts[i] <= c_ends[i]]
    MLu = SpinorState(zeros(ComplexF64, n_comp), zeros(ComplexF64, n_comp), zeros(ComplexF64, n_comp))

    return Workspace(
        g_out, 
        c_starts[active], 
        c_ends[active], 
        out_buffers[active], 
        Mu1, Mu0, Mum1, 
        res_u1, res_u0, res_um1, 
        u1_evals[active], u0_evals[active], um1_evals[active],
        cg_b, cg_x, cg_r, cg_r_new, cg_p, cg_Ap_out, cg_z,
        u_vec, out, sobolev_grad, ∇_A_N, ∇_A_M, u_proj, MLu
    )
end

function apply_A0_parallel!(
    y::Vector{ComplexF64}, x::Vector{ComplexF64}, 
    mesh, params, ws
)   
    chunk_starts = ws.chunk_starts 
    chunk_ends   = ws.chunk_ends 
    out_buffers  = ws.out_buffers 

    @threads for cid in 1:length(chunk_starts)
        fill!(out_buffers[cid], 0.0)
    end

    V_fun = params.V_fun
    T = mesh.T
    bn = mesh.bn
    Nd = mesh.x
    inner_indices = mesh.global_to_interior
    order = params.order_finite_element


    n_comp = length(x) ÷ 3

    no_of_basis = (order == 1) ? 3 : 6
    no_of_quad_points = 7

    w = @SVector [0.225, 0.132394152788506, 0.132394152788506, 0.132394152788506, 0.125939180544827, 0.125939180544827, 0.125939180544827]
    α = @SVector [0.059715871789770, 0.797426985353087]
    β = @SVector [0.470142064105115, 0.101286507323456]

    x_quad = (
        SVector(1/3, 1/3),
        SVector(β[1], β[1]), 
        SVector(α[1], β[1]), 
        SVector(β[1], α[1]),
        SVector(β[2], β[2]), 
        SVector(α[2], β[2]), 
        SVector(β[2], α[2])
    )

    phi = MMatrix{6, 7, Float64}(undef)
    grad_phi = MMatrix{2, 42, Float64}(undef)

    @inbounds for qpt in 1:no_of_quad_points
        xq1, xq2 = x_quad[qpt]
        if order == 1
            phi[1, qpt] = 1.0 - xq1 - xq2
            phi[2, qpt] = xq1
            phi[3, qpt] = xq2

            grad_phi[1, qpt] = -1.0; grad_phi[2, qpt] = -1.0
            grad_phi[1, 1*7 + qpt] =  1.0; grad_phi[2, 1*7 + qpt] =  0.0
            grad_phi[1, 2*7 + qpt] =  0.0; grad_phi[2, 2*7 + qpt] =  1.0
        else
            phi[1, qpt] = (1 - xq1 - xq2) * (1 - 2*xq1 - 2*xq2)
            phi[2, qpt] = 4 * xq1 * (1 - xq1 - xq2)
            phi[3, qpt] = xq1 * (2*xq1 - 1)
            phi[4, qpt] = 4 * xq1 * xq2
            phi[5, qpt] = xq2 * (2*xq2 - 1)
            phi[6, qpt] = 4 * xq2 * (1 - xq1 - xq2)

            grad_phi[1,  qpt] = 4*xq1 + 4*xq2 - 3; grad_phi[2,  qpt] = 4*xq1 + 4*xq2 - 3
            grad_phi[1, 1*7 + qpt] = -4*(2*xq1 + xq2 - 1); grad_phi[2, 1*7 + qpt] = -4*xq1
            grad_phi[1, 2*7 + qpt] = 4*xq1 - 1;        grad_phi[2, 2*7 + qpt] = 0.0
            grad_phi[1, 3*7 + qpt] = 4*xq2;            grad_phi[2, 3*7 + qpt] = 4*xq1
            grad_phi[1, 4*7 + qpt] = 0.0;            grad_phi[2, 4*7 + qpt] = 4*xq2 - 1
            grad_phi[1, 5*7 + qpt] = -4*xq2;           grad_phi[2, 5*7 + qpt] = -4*(xq1 + 2*xq2 - 1)
        end
    end

    @threads for cid in 1:length(chunk_starts)
        c_start = chunk_starts[cid]
        c_end   = chunk_ends[cid]
        y_loc   = out_buffers[cid]

        @inbounds for k in c_start:c_end
            tri1 = T[k, 1]
            tri3 = (order == 1) ? T[k, 2] : T[k, 3]
            tri5 = (order == 1) ? T[k, 3] : T[k, 5]

            z1 = SVector(Nd[tri1, 1], Nd[tri1, 2])
            z2 = SVector(Nd[tri3, 1], Nd[tri3, 2])
            z3 = SVector(Nd[tri5, 1], Nd[tri5, 2])

            J11 = z2[1] - z1[1]; J12 = z3[1] - z1[1]
            J21 = z2[2] - z1[2]; J22 = z3[2] - z1[2]

            detBT = 0.5 * (J11 * J22 - J12 * J21)
            invDet = 1.0 / (J11 * J22 - J12 * J21)

            iJ11 =  J22 * invDet; iJ12 = -J21 * invDet
            iJ21 = -J12 * invDet; iJ22 =  J11 * invDet

            for i in 1:no_of_basis
                node_i = T[k, i]
                bn[node_i] != 0 && continue
                eq_i = inner_indices[node_i]

                for j in 1:no_of_basis
                    node_j = T[k, j]
                    bn[node_j] != 0 && continue
                    eq_j = inner_indices[node_j]

                    base_ij = 0.0
                    mass_ij = 0.0

                    for qpt in 1:no_of_quad_points
                        phi_i = phi[i, qpt]
                        phi_j = phi[j, qpt]

                        gi_x = iJ11 * grad_phi[1, (i-1)*7 + qpt] + iJ12 * grad_phi[2, (i-1)*7 + qpt]
                        gi_y = iJ21 * grad_phi[1, (i-1)*7 + qpt] + iJ22 * grad_phi[2, (i-1)*7 + qpt]

                        gj_x = iJ11 * grad_phi[1, (j-1)*7 + qpt] + iJ12 * grad_phi[2, (j-1)*7 + qpt]
                        gj_y = iJ21 * grad_phi[1, (j-1)*7 + qpt] + iJ22 * grad_phi[2, (j-1)*7 + qpt]

                        xq, yq = x_quad[qpt]
                        x_phy = J11 * xq + J12 * yq + z1[1]
                        y_phy = J21 * xq + J22 * yq + z1[2]

                        m_q = 2.0 * phi_i * phi_j
                        stiff_q = (gi_x * gj_x + gi_y * gj_y)
                        pot_q = V_fun(SVector(x_phy, y_phy)) * m_q

                        base_ij += w[qpt] * (stiff_q + pot_q)
                        mass_ij += w[qpt] * m_q
                    end

                    base_ij *= detBT
                    mass_ij *= detBT

                    a0_u1  = base_ij + (params.q - params.p) * mass_ij
                    a0_u0  = base_ij 
                    a0_um1 = base_ij + (params.q + params.p) * mass_ij

                    y_loc[eq_i]            += a0_u1  * x[eq_j]
                    y_loc[n_comp + eq_i]   += a0_u0  * x[n_comp + eq_j] 
                    y_loc[2*n_comp + eq_i] += a0_um1 * x[2*n_comp + eq_j]
                end
            end
        end
    end

    num_chunks = length(chunk_starts)
    @threads for i in eachindex(y)
        s = 0.0
        @inbounds for cid in 1:num_chunks
            s += out_buffers[cid][i]
        end
        @inbounds y[i] = s
    end

    return nothing
end

function frechet_derivative!(
    u::SpinorState, mesh, params, ws::Workspace
)
    out_buffers  = ws.out_buffers 
    u_vec = ws.u_vec 
    out = ws.out
    g = ws.g_out

    um1 = u.um1 
    u0  = u.u0 
    u1  = u.u1 

    x   = mesh.x 
    T   = mesh.T 
    bn  = mesh.bn
    global_to_interior = mesh.global_to_interior

    order_finite_element = params.order_finite_element
    β0 = params.β0
    β1 = params.β1

    to_vec!(u_vec, u)

    for buf in out_buffers
        fill!(buf, 0.0)
    end
    fill!(out, 0.0)

    apply_A0_parallel!(out, u_vec, mesh, params, ws)
    from_vec!(g, out)

    g1  = g.u1
    g0  = g.u0
    gm1 = g.um1

    # 2. Static Quadrature Constants
    w = @SVector [0.225, 0.132394152788506, 0.132394152788506, 0.132394152788506, 0.125939180544827, 0.125939180544827, 0.125939180544827]
    α = @SVector [0.059715871789770, 0.797426985353087]
    β = @SVector [0.470142064105115, 0.101286507323456]
    x_quad = (
        SVector(1/3, 1/3),
        SVector(β[1], β[1]),
        SVector(α[1], β[1]),
        SVector(β[1], α[1]),
        SVector(β[2], β[2]),
        SVector(α[2], β[2]),
        SVector(β[2], α[2])
    )
    no_of_quad_points = 7
    no_of_basis = (order_finite_element == 1) ? 3 : 6

    # 3. Main Assembly Loop (0 allocations)
    @inbounds for k in axes(T, 1)
        # Check interior nodes without allocating an Int[] array
        has_interior = false
        for a_idx in 1:3
            if bn[T[k, a_idx]] == 0
                has_interior = true
                break
            end
        end
        !has_interior && continue

        # Node coordinates
        a1 = T[k, 1]
        z1 = SVector(x[a1, 1], x[a1, 2])

        if order_finite_element == 1
            a2, a3 = T[k, 2], T[k, 3]
            z2 = SVector(x[a2, 1], x[a2, 2])
            z3 = SVector(x[a3, 1], x[a3, 2])
        else
            z2 = SVector(x[T[k, 3], 1], x[T[k, 3], 2]) 
            z3 = SVector(x[T[k, 5], 1], x[T[k, 5], 2]) 
        end

        detBT = 0.5 * ((z2[1]-z1[1]) * (z3[2]-z1[2]) - (z3[1]-z1[1]) * (z2[2]-z1[2]))

        um1_loc = MVector{6,ComplexF64}(undef)
        u0_loc  = MVector{6,ComplexF64}(undef)
        u1_loc  = MVector{6,ComplexF64}(undef)
        fill!(um1_loc, 0.0im)
        fill!(u0_loc, 0.0im)
        fill!(u1_loc, 0.0im)

        for i in 1:no_of_basis
            node = T[k, i]
            if bn[node] == 0
                gi = global_to_interior[node]
                um1_loc[i] = um1[gi]
                u0_loc[i]  = u0[gi]
                u1_loc[i]  = u1[gi]
            end
        end

        phi_quad = NTuple{7, SVector{6, Float64}}(
        begin
            xi1, xi2 = x_quad[qpt]

            if order_finite_element == 1
                SVector(
                    1.0 - xi1 - xi2,
                    xi1,
                    xi2,
                    0.0, 0.0, 0.0
                )
            else
                SVector(
                    (1.0 - xi1 - xi2) * (1.0 - 2.0 * xi1 - 2.0 * xi2),
                    4.0 * xi1 * (1.0 - xi1 - xi2),
                    xi1 * (2.0 * xi1 - 1.0),
                    4.0 * xi1 * xi2,
                    xi2 * (2.0 * xi2 - 1.0),
                    4.0 * xi2 * (1.0 - xi1 - xi2)
                )
            end
        end for qpt in 1:7
          )

        nlm1_loc = MVector{6,ComplexF64}(undef)
        nl0_loc  = MVector{6,ComplexF64}(undef)
        nl1_loc  = MVector{6,ComplexF64}(undef)
        fill!(nlm1_loc, 0.0im)
        fill!(nl0_loc, 0.0im)
        fill!(nl1_loc, 0.0im)

        for qpt in 1:no_of_quad_points
            wq = w[qpt]
            phi = phi_quad[qpt]

            uqm1 = 0.0im
            uq0  = 0.0im
            uq1  = 0.0im

            for i in 1:no_of_basis
                    uqm1 += phi[i] * um1_loc[i]
                    uq0  += phi[i] * u0_loc[i]
                    uq1  += phi[i] * u1_loc[i]
            end

            abs_um1_sq = abs2(uqm1)
            abs_u0_sq  = abs2(uq0)
            abs_u1_sq  = abs2(uq1)

            rho = abs_um1_sq + abs_u0_sq + abs_u1_sq
            
            f_rho_m1 = 2 * β0 * rho * uqm1
            f_rho_0  = 2 * β0 * rho * uq0
            f_rho_1  = 2 * β0 * rho * uq1

            Δ = abs_u1_sq - abs_um1_sq
            A = conj(uqm1) * uq0 + conj(uq0) * uq1
            B = conj(A)

            f_spin_m1 = 2 * β1 * (-Δ * uqm1 + B * uq0)
            f_spin_0  = 2 * β1 * ( B * uq1  + A * uqm1)
            f_spin_1  = 2 * β1 * ( Δ * uq1  + A * uq0)

            fm1 = f_rho_m1 + f_spin_m1
            f0  = f_rho_0  + f_spin_0
            f1  = f_rho_1  + f_spin_1

            weight = wq * detBT
            for i in 1:no_of_basis
                nlm1_loc[i] += weight * phi[i] * fm1
                nl0_loc[i]  += weight * phi[i] * f0
                nl1_loc[i]  += weight * phi[i] * f1
            end
        end

        for i in 1:no_of_basis
            node = T[k, i]
            if bn[node] == 0
                gi = global_to_interior[node]
                gm1[gi] += nlm1_loc[i]
                g0[gi]  += nl0_loc[i]
                g1[gi]  += nl1_loc[i]
            end
        end
    end

    return g
end

function get_residual!(u, params, M, E_prime, ws)
    C_mag = params.C_mag
    Mu1  = ws.Mu1 
    Mu0  = ws.Mu0 
    Mum1 = ws.Mum1

    res_u1  = ws.res_u1
    res_u0  = ws.res_u0
    res_um1 = ws.res_um1

    u1  = u.u1
    u0  = u.u0
    um1 = u.um1

    mul!(Mu1,  M, u1)
    mul!(Mu0,  M, u0)
    mul!(Mum1, M, um1)


    b1 = real(dot(E_prime.u1, u1) + #<E'(u),u> 
          dot(E_prime.u0, u0) +
          dot(E_prime.um1, um1))

    b2 = real(dot(E_prime.u1, u1) - #<E'(u),Lu>
          dot(E_prime.um1, um1))

    LuLu = real(dot(u1, Mu1) + dot(um1, Mum1))

    det_inv = 1.0 / (LuLu - C_mag^2)

    μ = det_inv * (LuLu * b1 - C_mag * b2)
    λ = det_inv * (b2 - C_mag * b1)

    res_u1  .= E_prime.u1  .- (μ + λ) .* Mu1
    res_u0  .= E_prime.u0  .- μ       .* Mu0
    res_um1 .= E_prime.um1 .- (μ - λ) .* Mum1

    res = sqrt(real(
        dot(res_u1, M, res_u1) + dot(res_u0, M, res_u0) + dot(res_um1, M, res_um1)
    ))

    return res, μ, λ, SpinorState(Mu1, Mu0, Mum1)
end

function apply_A_with_spin_parallel!(
    out::Vector{ComplexF64}, vec::Vector{ComplexF64}, 
    u1, u0, um1, mesh, params, n_comp::Int, no_of_basis::Int, 
     w::AbstractVector{Float64}, phi_in_quad_point::AbstractMatrix{Float64}, 
    β0::Float64, β1::Float64,
    ws::Workspace
)   
    chunk_starts = ws.chunk_starts 
    chunk_ends   = ws.chunk_ends 
    out_buffers  = ws.out_buffers 
    num_chunks = length(chunk_starts)
    u1_evals   = [Vector{ComplexF64}(undef, no_of_basis) for _ in 1:num_chunks]
    u0_evals   = [Vector{ComplexF64}(undef, no_of_basis) for _ in 1:num_chunks]
    um1_evals  = [Vector{ComplexF64}(undef, no_of_basis) for _ in 1:num_chunks]

    apply_A0_parallel!(out, vec, mesh, params, ws)

    # Reset thread-local accumulators
    @threads for cid in 1:length(chunk_starts)
        fill!(out_buffers[cid], 0.0)
    end

    x_nodes = mesh.x
    T = mesh.T
    bn = mesh.bn
    g2i = mesh.global_to_interior

    # Parallel Element Loop
    @threads for cid in 1:length(chunk_starts)
        c_start  = chunk_starts[cid]
        c_end    = chunk_ends[cid]
        out_loc  = out_buffers[cid]
        u1_eval  = u1_evals[cid]
        u0_eval  = u0_evals[cid]
        um1_eval = um1_evals[cid]

        @inbounds for k in c_start:c_end
            tri1, tri2, tri3 = T[k, 1], T[k, 2], T[k, 3]
            z1_x, z1_y = x_nodes[tri1, 1], x_nodes[tri1, 2]
            
            if no_of_basis == 3
                z2_x, z2_y = x_nodes[tri2, 1], x_nodes[tri2, 2]
                z3_x, z3_y = x_nodes[tri3, 1], x_nodes[tri3, 2]
            else
                z2_x, z2_y = x_nodes[T[k, 3], 1], x_nodes[T[k, 3], 2] 
                z3_x, z3_y = x_nodes[T[k, 5], 1], x_nodes[T[k, 5], 2] 
            end

            detBT = 0.5 * ((z2_x - z1_x) * (z3_y - z1_y) - (z3_x - z1_x) * (z2_y - z1_y))

            for i in 1:no_of_basis
                node = T[k, i]
                if bn[node] == 0
                    idx = g2i[node]
                    u1_eval[i]  = ComplexF64(u1[idx])
                    u0_eval[i]  = ComplexF64(u0[idx])
                    um1_eval[i] = ComplexF64(um1[idx])
                else
                    u1_eval[i]  = 0.0im
                    u0_eval[i]  = 0.0im
                    um1_eval[i] = 0.0im
                end
            end

            A_q   = MVector{7, ComplexF64}(undef)
            Fz    = MVector{7, Float64}(undef)
            rho_q = MVector{7, Float64}(undef)

            for l in 1:7
                s_u1, s_u0, s_um1 = 0.0im, 0.0im, 0.0im
                for b in 1:no_of_basis
                    p_val = phi_in_quad_point[b, l]
                    s_u1  += p_val * u1_eval[b]
                    s_u0  += p_val * u0_eval[b]
                    s_um1 += p_val * um1_eval[b]
                end
                
                rho_q[l] = abs2(s_u1) + abs2(s_u0) + abs2(s_um1)
                
                A_q[l]   = s_u0 * conj(s_um1) + s_u1 * conj(s_u0)
                Fz[l]    = real(abs2(s_u1) - abs2(s_um1))
            end

            for j in 1:no_of_basis
                node_j = T[k, j]
                idx_j = g2i[node_j]
                
                if idx_j > 0 && bn[node_j] == 0
                    v_j1  = ComplexF64(vec[idx_j])
                    v_j0  = ComplexF64(vec[idx_j + n_comp])
                    v_jm1 = ComplexF64(vec[idx_j + 2*n_comp])

                    for i in 1:no_of_basis
                        node_i = T[k, i]
                        idx_i = g2i[node_i]
                        
                        if idx_i > 0 && bn[node_i] == 0
                            out_spin_1  = 0.0im
                            out_spin_0  = 0.0im
                            out_spin_m1 = 0.0im

                            for l in 1:7
                                fac = phi_in_quad_point[i, l] * phi_in_quad_point[j, l]
                                fz  = Fz[l]
                                rho = rho_q[l]
                                A   = A_q[l]
                                A_bar = conj(A)


                                coeff_β0 = 2.0 * β0 * rho
                                b0_1  = coeff_β0 * v_j1
                                b0_0  = coeff_β0 * v_j0
                                b0_m1 = coeff_β0 * v_jm1

                                b1_1  = 2.0 * β1 * (fz * v_j1 + A * v_j0)
                                b1_0  = 2.0 * β1 * (A_bar * v_j1 + A * v_jm1)
                                b1_m1 = 2.0 * β1 * (A_bar * v_j0 - fz * v_jm1)

                                weight = w[l] * fac * detBT
                                out_spin_1  += weight * (b0_1  + b1_1)
                                out_spin_0  += weight * (b0_0  + b1_0)
                                out_spin_m1 += weight * (b0_m1 + b1_m1)
                            end

                            out_loc[idx_i]                          += out_spin_1
                            out_loc[idx_i + n_comp]                 += out_spin_0
                            out_loc[idx_i + 2*n_comp]               += out_spin_m1
                        end
                    end
                end
            end
        end
    end

    @threads for i in eachindex(out)
        s = 0.0
        @inbounds for cid in 1:num_chunks
            s += out_buffers[cid][i]
        end
        @inbounds out[i] += s
    end

    return nothing
end

function apply_A_spin_diagonal_parallel!(
    out::Vector{ComplexF64}, vec::Vector{ComplexF64}, 
    u1, u0, um1, mesh, params, n_comp::Int, no_of_basis::Int, 
     w::AbstractVector{Float64}, phi_in_quad_point::AbstractMatrix{Float64}, 
    β0::Float64, β1::Float64,
    ws::Workspace
)   
    chunk_starts = ws.chunk_starts 
    chunk_ends   = ws.chunk_ends 
    out_buffers  = ws.out_buffers 
     num_chunks = length(chunk_starts)
    u1_evals   = [Vector{ComplexF64}(undef, no_of_basis) for _ in 1:num_chunks]
    u0_evals   = [Vector{ComplexF64}(undef, no_of_basis) for _ in 1:num_chunks]
    um1_evals  = [Vector{ComplexF64}(undef, no_of_basis) for _ in 1:num_chunks]

    apply_A0_parallel!(out, vec, mesh, params, ws)

    # Reset thread-local accumulators
    @threads for cid in 1:length(chunk_starts)
        fill!(out_buffers[cid], 0.0)
    end

    x_nodes = mesh.x
    T = mesh.T
    bn = mesh.bn
    g2i = mesh.global_to_interior

    # Parallel Element Loop
    @threads for cid in 1:length(chunk_starts)
        c_start  = chunk_starts[cid]
        c_end    = chunk_ends[cid]
        out_loc  = out_buffers[cid]
        u1_eval  = u1_evals[cid]
        u0_eval  = u0_evals[cid]
        um1_eval = um1_evals[cid]

        @inbounds for k in c_start:c_end
            tri1, tri2, tri3 = T[k, 1], T[k, 2], T[k, 3]
            z1_x, z1_y = x_nodes[tri1, 1], x_nodes[tri1, 2]
            
            if no_of_basis == 3
                z2_x, z2_y = x_nodes[tri2, 1], x_nodes[tri2, 2]
                z3_x, z3_y = x_nodes[tri3, 1], x_nodes[tri3, 2]
            else
                z2_x, z2_y = x_nodes[T[k, 3], 1], x_nodes[T[k, 3], 2] 
                z3_x, z3_y = x_nodes[T[k, 5], 1], x_nodes[T[k, 5], 2] 
            end

            detBT = 0.5 * ((z2_x - z1_x) * (z3_y - z1_y) - (z3_x - z1_x) * (z2_y - z1_y))

            for i in 1:no_of_basis
                node = T[k, i]
                if bn[node] == 0
                    idx = g2i[node]
                    u1_eval[i]  = ComplexF64(u1[idx])
                    u0_eval[i]  = ComplexF64(u0[idx])
                    um1_eval[i] = ComplexF64(um1[idx])
                else
                    u1_eval[i]  = 0.0im
                    u0_eval[i]  = 0.0im
                    um1_eval[i] = 0.0im
                end
            end

            rho_q      = MVector{7, Float64}(undef)
            abs2_u1_q  = MVector{7, Float64}(undef)
            abs2_u0_q  = MVector{7, Float64}(undef)
            abs2_um1_q = MVector{7, Float64}(undef)

            for l in 1:7
                s_u1, s_u0, s_um1 = 0.0im, 0.0im, 0.0im
                for b in 1:no_of_basis
                    p_val = phi_in_quad_point[b, l]
                    s_u1  += p_val * u1_eval[b]
                    s_u0  += p_val * u0_eval[b]
                    s_um1 += p_val * um1_eval[b]
                end
                
                abs2_u1_q[l]  = abs2(s_u1)
                abs2_u0_q[l]  = abs2(s_u0)
                abs2_um1_q[l] = abs2(s_um1)
                rho_q[l]      = abs2_u1_q[l] + abs2_u0_q[l] + abs2_um1_q[l]
            end

            for j in 1:no_of_basis
                node_j = T[k, j]
                idx_j = g2i[node_j]
                
                if idx_j > 0 && bn[node_j] == 0
                    v_j1  = ComplexF64(vec[idx_j])
                    v_j0  = ComplexF64(vec[idx_j + n_comp])
                    v_jm1 = ComplexF64(vec[idx_j + 2*n_comp])

                    for i in 1:no_of_basis
                        node_i = T[k, i]
                        idx_i = g2i[node_i]
                        
                        if idx_i > 0 && bn[node_i] == 0
                            out1  = 0.0im
                            out0  = 0.0im
                            outm1 = 0.0im

                            for l in 1:7
                                fac = phi_in_quad_point[i, l] * phi_in_quad_point[j, l]
                                rho = rho_q[l]
                                abs2_u1  = abs2_u1_q[l]
                                abs2_u0  = abs2_u0_q[l]
                                abs2_um1 = abs2_um1_q[l]

                                coeff_β0 = 2.0 * β0 * rho
                                b0_1  = coeff_β0 * v_j1
                                b0_0  = coeff_β0 * v_j0
                                b0_m1 = coeff_β0 * v_jm1

                                b1_1  = 2.0 * β1 * (abs2_u0 + abs2_u1 - abs2_um1) * v_j1
                                b1_0  = 2.0 * β1 * (abs2_u1 + abs2_um1) * v_j0
                                b1_m1 = 2.0 * β1 * (abs2_u0 + abs2_um1 - abs2_u1) * v_jm1

                                weight = w[l] * fac * detBT
                                out1  += weight * (b0_1 + b1_1)
                                out0  += weight * (b0_0 + b1_0)
                                outm1 += weight * (b0_m1 + b1_m1) 
                            end

                            out_loc[idx_i]                          += out1
                            out_loc[idx_i + n_comp]                 += out0
                            out_loc[idx_i + 2*n_comp]               += outm1
                        end
                    end
                end
            end
        end
    end

    @threads for i in eachindex(out)
        s = 0.0
        @inbounds for cid in 1:num_chunks
            s += out_buffers[cid][i]
        end
        @inbounds out[i] += s
    end

    return nothing
end

function apply_A_density_parallel!(
    out::Vector{ComplexF64}, vec::Vector{ComplexF64}, 
    u1, u0, um1, mesh, params, n_comp::Int, no_of_basis::Int, 
     w::AbstractVector{Float64}, phi_in_quad_point::AbstractMatrix{Float64}, 
    β0::Float64, β1::Float64,
    ws::Workspace
)   
    chunk_starts = ws.chunk_starts 
    chunk_ends   = ws.chunk_ends 
    out_buffers  = ws.out_buffers 
     num_chunks = length(chunk_starts)
    u1_evals   = [Vector{ComplexF64}(undef, no_of_basis) for _ in 1:num_chunks]
    u0_evals   = [Vector{ComplexF64}(undef, no_of_basis) for _ in 1:num_chunks]
    um1_evals  = [Vector{ComplexF64}(undef, no_of_basis) for _ in 1:num_chunks]

    apply_A0_parallel!(out, vec, mesh, params, ws)

    # Reset thread-local accumulators
    @threads for cid in 1:length(chunk_starts)
        fill!(out_buffers[cid], 0.0)
    end

    x_nodes = mesh.x
    T = mesh.T
    bn = mesh.bn
    g2i = mesh.global_to_interior

    # Parallel Element Loop
    @threads for cid in 1:length(chunk_starts)
        c_start  = chunk_starts[cid]
        c_end    = chunk_ends[cid]
        out_loc  = out_buffers[cid]
        u1_eval  = u1_evals[cid]
        u0_eval  = u0_evals[cid]
        um1_eval = um1_evals[cid]

        @inbounds for k in c_start:c_end
            tri1, tri2, tri3 = T[k, 1], T[k, 2], T[k, 3]
            z1_x, z1_y = x_nodes[tri1, 1], x_nodes[tri1, 2]
            
            if no_of_basis == 3
                z2_x, z2_y = x_nodes[tri2, 1], x_nodes[tri2, 2]
                z3_x, z3_y = x_nodes[tri3, 1], x_nodes[tri3, 2]
            else
                z2_x, z2_y = x_nodes[T[k, 3], 1], x_nodes[T[k, 3], 2] 
                z3_x, z3_y = x_nodes[T[k, 5], 1], x_nodes[T[k, 5], 2] 
            end

            detBT = 0.5 * ((z2_x - z1_x) * (z3_y - z1_y) - (z3_x - z1_x) * (z2_y - z1_y))

            for i in 1:no_of_basis
                node = T[k, i]
                if bn[node] == 0
                    idx = g2i[node]
                    u1_eval[i]  = ComplexF64(u1[idx])
                    u0_eval[i]  = ComplexF64(u0[idx])
                    um1_eval[i] = ComplexF64(um1[idx])
                else
                    u1_eval[i]  = 0.0im
                    u0_eval[i]  = 0.0im
                    um1_eval[i] = 0.0im
                end
            end

            rho_q      = MVector{7, Float64}(undef)

            for l in 1:7
                s_u1, s_u0, s_um1 = 0.0im, 0.0im, 0.0im
                for b in 1:no_of_basis
                    p_val = phi_in_quad_point[b, l]
                    s_u1  += p_val * u1_eval[b]
                    s_u0  += p_val * u0_eval[b]
                    s_um1 += p_val * um1_eval[b]
                end

                rho_q[l]      = abs2(s_u1) + abs2(s_u0) + abs2(s_um1)
            end

            for j in 1:no_of_basis
                node_j = T[k, j]
                idx_j = g2i[node_j]
                
                if idx_j > 0 && bn[node_j] == 0
                    v_j1  = ComplexF64(vec[idx_j])
                    v_j0  = ComplexF64(vec[idx_j + n_comp])
                    v_jm1 = ComplexF64(vec[idx_j + 2*n_comp])

                    for i in 1:no_of_basis
                        node_i = T[k, i]
                        idx_i = g2i[node_i]
                        
                        if idx_i > 0 && bn[node_i] == 0
                            out1  = 0.0im
                            out0  = 0.0im
                            outm1 = 0.0im

                            for l in 1:7
                                fac = phi_in_quad_point[i, l] * phi_in_quad_point[j, l]
                                rho = rho_q[l]

                                coeff_β0 = 2.0 * β0 * rho
                                b0_1  = coeff_β0 * v_j1
                                b0_0  = coeff_β0 * v_j0
                                b0_m1 = coeff_β0 * v_jm1

                                weight = w[l] * fac * detBT
                                out1  += weight * b0_1 
                                out0  += weight * b0_0 
                                outm1 += weight * b0_m1 
                            end

                            out_loc[idx_i]                          += out1
                            out_loc[idx_i + n_comp]                 += out0
                            out_loc[idx_i + 2*n_comp]               += outm1
                        end
                    end
                end
            end
        end
    end

    @threads for i in eachindex(out)
        s = 0.0
        @inbounds for cid in 1:num_chunks
            s += out_buffers[cid][i]
        end
        @inbounds out[i] += s
    end

    return nothing
end

function apply_A_parallel!(
    out::Vector{ComplexF64}, vec::Vector{ComplexF64}, 
    u1, u0, um1, mesh, params, n_comp::Int, no_of_basis::Int, 
     w::AbstractVector{Float64}, phi_in_quad_point::AbstractMatrix{Float64}, 
    β0::Float64, β1::Float64,
    ws::Workspace, metric
)   
   if metric == "spin"
        apply_A_with_spin_parallel!(
                out, vec, u1, u0, um1, mesh, params,
                n_comp, no_of_basis, w, phi_in_quad_point, β0, β1,
                ws
            )
   elseif metric == "spin-diagonal"
        apply_A_spin_diagonal_parallel!(
                out, vec, u1, u0, um1, mesh, params,
                n_comp, no_of_basis, w, phi_in_quad_point, β0, β1,
                ws
            )
   elseif metric == "density"
        apply_A_density_parallel!(
                out, vec, u1, u0, um1, mesh, params,
                n_comp, no_of_basis, w, phi_in_quad_point, β0, β1,
                ws
            )
   elseif metric == "A0"
        apply_A0_parallel!(
                    out, vec, mesh, params, ws
                )
   end

    return nothing
end

function cg_A_parallel!( #output x = A\b
    x_state::SpinorState,   # Pre-allocated output SpinorState
    mesh, params, u, b_state, max_iter, tol, metric, P, ws::Workspace
)
    β0 = params.β0 
    β1 = params.β1
    
    n_comp = length(u.um1)
    n = 3 * n_comp
    
    b      = ws.cg_b
    x      = ws.cg_x
    r      = ws.cg_r
    r_new  = ws.cg_r_new
    p      = ws.cg_p
    Ap_out = ws.cg_Ap_out
    z      = ws.cg_z

    to_vec!(b, b_state)
    fill!(x, 0.0)
    copyto!(r, b)

    r_norm = norm(r)
    if r_norm < tol
        from_vec!(x_state, x)
        return x_state
    end

    # Preconditioning step : z_0 = P \ r_0
    ldiv!(z, P, r)
    copyto!(p, z)
    rz = dot(r, z)

    w = @SVector [0.225, 0.132394152788506, 0.132394152788506, 0.132394152788506, 0.125939180544827, 0.125939180544827, 0.125939180544827]
    α = @SVector [0.059715871789770, 0.797426985353087]
    β = @SVector [0.470142064105115, 0.101286507323456]
    
    x_quad = (
        SVector(1/3, 1/3),
        SVector(β[1], β[1]), 
        SVector(α[1], β[1]), 
        SVector(β[1], α[1]),  
        SVector(β[2], β[2]), 
        SVector(α[2], β[2]), 
        SVector(β[2], α[2])
    )
    no_of_quad_points = 7

    order = params.order_finite_element
    no_of_basis = (order == 1) ? 3 : 6

    phi_in_quad_point = MMatrix{6, 7, Float64}(undef)
    for qpt in 1:no_of_quad_points
        xq1, xq2 = x_quad[qpt][1], x_quad[qpt][2]
        if order == 1
            phi_in_quad_point[1, qpt] = -xq1 - xq2 + 1.0
            phi_in_quad_point[2, qpt] = xq1
            phi_in_quad_point[3, qpt] = xq2
        else
            phi_in_quad_point[1, qpt] = (1 - xq1 - xq2) * (1 - 2*xq1 - 2*xq2)
            phi_in_quad_point[2, qpt] = 4 * xq1 * (1 - xq1 - xq2)
            phi_in_quad_point[3, qpt] = xq1 * (2*xq1 - 1)
            phi_in_quad_point[4, qpt] = 4 * xq1 * xq2
            phi_in_quad_point[5, qpt] = xq2 * (2*xq2 - 1)
            phi_in_quad_point[6, qpt] = 4 * xq2 * (1 - xq1 - xq2)
        end
    end

    um1, u0, u1 = u.um1, u.u0, u.u1

    k = 0
    @inbounds while r_norm >= tol && k < max_iter
        apply_A_parallel!(
                Ap_out, p, u1, u0, um1, mesh, params,
                n_comp, no_of_basis, w, phi_in_quad_point, β0, β1,
                ws, metric
            )
              
        p_Ap = dot(p, Ap_out)
        α_k = rz / p_Ap

        @simd for i in 1:n
            x[i]     += α_k * p[i]
            r_new[i]  = r[i] - α_k * Ap_out[i]
        end

        r_norm = norm(r_new)
        if r_norm < tol
            break
        end      

        # Compute z_{k+1} = P \ r_{k+1}
        ldiv!(z, P, r_new)

        rz_new = dot(r_new, z)
        β_k = rz_new / rz

        @simd for i in 1:n
            p[i] = z[i] + β_k * p[i]
            r[i] = r_new[i]
        end

        rz = rz_new
        k += 1
    end
    
    if r_norm >= tol
        error("CG did not converge within $max_iter iterations (res_norm = $r_norm)")
    end

    from_vec!(x_state, x)
    return x_state
end

function get_Riemannian_gradient(u, Mu, E_prime, mesh, params, metric, P, res, ws::Workspace)
    # ∇_A_N = A\Mu
    ws.MLu.u1 .= Mu.u1 
    ws.MLu.um1 .= -Mu.um1
    MLu = ws.MLu
    
    # tol = min(1e-6, 0.1*res) #for adaptive tolerance
    tol = 1e-12

    if metric == "spin"
        sobolev_grad = u 
    else     
        sobolev_grad = cg_A_parallel!(ws.sobolev_grad, mesh, params, u, E_prime, 1000, tol, metric, P, ws)
    end

    ∇_A_N  = cg_A_parallel!(ws.∇_A_N, mesh, params, u, Mu, 1000, tol, metric, P, ws)
    if metric == "spin"
        ∇_A_M  = cg_A_parallel!(ws.∇_A_M, mesh, params, u, ws.MLu, 1000, tol, metric, P, ws)
    else 
        #The other metrics are diagonal, so we can make use of that structure
        ws.∇_A_M.u1  .=  ∇_A_N.u1
        ws.∇_A_M.um1 .= -∇_A_N.um1
        ∇_A_M = ws.∇_A_M
    end

    bN = real(
        dot(E_prime.u1, ∇_A_N.u1) +
        dot(E_prime.u0, ∇_A_N.u0) +
        dot(E_prime.um1, ∇_A_N.um1)
    )

    bM = real(
        dot(E_prime.u1, ∇_A_M.u1) +
        dot(E_prime.u0, ∇_A_M.u0) +
        dot(E_prime.um1, ∇_A_M.um1)
    )

    NN = real(
        dot(Mu.u1, ∇_A_N.u1) +
        dot(Mu.u0, ∇_A_N.u0) +
        dot(Mu.um1, ∇_A_N.um1)
    )

    NM = real(
        dot(Mu.u1, ∇_A_M.u1) +
        dot(Mu.u0, ∇_A_M.u0) +
        dot(Mu.um1, ∇_A_M.um1)
    )

    MN = real(
        dot(MLu.u1, ∇_A_N.u1) +
        dot(MLu.u0, ∇_A_N.u0) +
        dot(MLu.um1, ∇_A_N.um1)
    )

    MM = real(
        dot(MLu.u1, ∇_A_M.u1) +
        dot(MLu.u0, ∇_A_M.u0) +
        dot(MLu.um1, ∇_A_M.um1)
    )

    G = [
        NN NM
        MN MM
    ]

    rhs = [bN, bM]

    μ_A, λ_A = G \ rhs


    grad = SpinorState(
        sobolev_grad.u1 - μ_A * ∇_A_N.u1 - λ_A * ∇_A_M.u1,
        sobolev_grad.u0 - μ_A * ∇_A_N.u0 - λ_A * ∇_A_M.u0,
        sobolev_grad.um1 - μ_A * ∇_A_N.um1 - λ_A * ∇_A_M.um1
    )

    return grad, μ_A, λ_A
 
end

function get_retraction!(
    u_out::SpinorState,
    u::SpinorState, 
    dn::SpinorState, 
    τ::Float64, 
    M_final, 
    C_mag::Float64, 
    retraction::String
)

    @. u_out.u1  = u.u1  + τ * dn.u1
    @. u_out.u0  = u.u0  + τ * dn.u0
    @. u_out.um1 = u.um1 + τ * dn.um1

    N_u1  = real(dot(u_out.u1,  M_final, u_out.u1))
    N_u0  = real(dot(u_out.u0,  M_final, u_out.u0))
    N_um1 = real(dot(u_out.um1, M_final, u_out.um1))

    if retraction == "newton" # L2 normal retraction
        LuLu          = real(dot(u.um1, M_final, u.um1) + dot(u.u1, M_final, u.u1)) # (Lu,Lu)
        dot_Lu_out_Lu         = real(dot(u_out.u1,  M_final, u.u1)  + dot(u_out.um1, M_final, u.um1)) # (Lu_out, Lu)

        u_out_sq = real(N_um1 + N_u0 + N_u1) #(u_out, u_out)
        dot_Lu_out_u_out= real(N_u1 - N_um1) # (Lu_out, u_out)

        if LuLu-C_mag^2 < -1e-12 * max(abs(LuLu), abs(C_mag^2), 1.0)
            @show(LuLu-C_mag^2)
            error("Regularity check failed")
        end

        α = 0.0
        β = 0.0

        rhs_N = 10.0
        rhs_M = 10.0
        count = 1

        while (abs(rhs_N) > 1e-12 || abs(rhs_M) > 1e-12) && count <= 5000
            rhs_N = α^2 + β^2 * LuLu - 2.0 * α - 2.0 * β * C_mag + 2.0 * α * β * C_mag + u_out_sq - 1.0
            rhs_M = α^2 * C_mag + β^2 * C_mag - 2.0 * α * C_mag - 2.0 * β * dot_Lu_out_Lu + 2.0 * α * β * LuLu + dot_Lu_out_u_out - C_mag #only holds for spin1 (β^2 term)

            partial_N_α = 2.0 * α  + 2.0 * β * C_mag - 2.0
            partial_N_β = 2.0 * β * LuLu  + 2.0 * α * C_mag - 2.0 * C_mag

            partial_M_α = 2.0 * α * C_mag + 2.0 * β * LuLu - 2.0 * C_mag
            partial_M_β = 2.0 * β * C_mag + 2.0 * α * LuLu - 2.0 * dot_Lu_out_Lu

            detJ = (partial_N_α * partial_M_β - partial_N_β * partial_M_α)
            det_inv = 1.0 / detJ

            α -= det_inv * (partial_M_β * rhs_N - partial_N_β * rhs_M)
            β -= det_inv * (partial_N_α * rhs_M - partial_M_α * rhs_N)
            count += 1       
        end

        @. u_out.um1 -= (α - β) * u.um1
        @. u_out.u0  -= α * u.u0
        @. u_out.u1  -= (α + β) * u.u1
        
    elseif retraction == "inexact"
       
        m0 = N_u1 + N_u0 + N_um1 
        m1 = N_u1 - N_um1 
        m2 = N_u1 + N_um1

        a = (m0 * (C_mag * m0 - m1)) / (2.0 * (m0 * m2 - m1^2))
        denom = m0 + 2.0 * m1 * a + m2 * a^2
        if denom < 0 
            error(printf("Inexact projection not possible"))
        end
        σ0 = 1.0 / sqrt(denom)
        σ1  = σ0 * (1.0 + a)
        σm1 = σ0 * (1.0 - a)

        @. u_out.u1  *= σ1
        @. u_out.u0  *= σ0
        @. u_out.um1 *= σm1
        N_final = real(
    dot(u_out.u1, M_final, u_out.u1) +
    dot(u_out.u0, M_final, u_out.u0) +
    dot(u_out.um1, M_final, u_out.um1)
)

M_final_actual = real(
    dot(u_out.u1, M_final, u_out.u1) -
    dot(u_out.um1, M_final, u_out.um1)
)

@show N_final - 1.0
@show M_final_actual - C_mag

    elseif retraction == "exact"
        a2 = (1.0 - C_mag) * N_u1 
        a1 = -C_mag * N_u0 
        a0 = -(1.0 + C_mag) * N_um1

        if abs(a2) < 1e-14 
            if abs(a1) < 1e-14
                error("Exact projection not possible: a2 and a1 are both zero")
            end
            γ = -a0 / a1
        else
            p = a1 / a2 
            q = a0 / a2
            factor = (p / 2.0)^2 - q 
            if factor <= 0.0 
                error("Exact projection not possible: negative discriminant")
            end
            γ = -p / 2.0 + sqrt(factor)
        end

        if γ <= 0.0 
            error("Exact projection not possible: no positive root found for γ")
        end

        denom = sqrt(N_um1 / γ + N_u0 + γ * N_u1)
        σ0  = 1.0 / denom
        σ1  = sqrt(γ) * σ0     
        σm1 = σ0 / sqrt(γ)

        @. u_out.u1  *= σ1
        @. u_out.u0  *= σ0
        @. u_out.um1 *= σm1

    elseif retraction == "closed-form"
        u_out_norm_sq = N_um1 + N_u0 + N_u1
        Lu_out_norm_sq = N_u1 + N_um1
        u_out_Lu_out = N_u1 - N_um1

        denom = u_out_norm_sq * Lu_out_norm_sq - u_out_Lu_out^2
        if denom <0 
            @show(denom)
        end

        σ_nom = zeros(3)

        for j in 1:3
            l = 2 - j  # 2 = (F+1)/2

            num = Lu_out_norm_sq -
                (C_mag + l) * u_out_Lu_out +
                C_mag * l * u_out_norm_sq
            if num < 0
                @show(num)
            end

            σ_nom[j] = sqrt(num / denom)
        end

        @. u_out.u1  *= σ_nom[1]
        @. u_out.u0  *= σ_nom[2]
        @. u_out.um1 *= σ_nom[3]


    end

    return nothing
end

function optimal_tau_blackbox(
    u::SpinorState, dn::SpinorState, u_proj::SpinorState, 
    mesh, params, M, spin_mats, 
    a::Float64, b::Float64, retraction::String
)
    C_mag = params.C_mag

    function f(τ_val)
        get_retraction!(u_proj, u, dn, τ_val, M, C_mag, retraction)
        return spin1_energy(u_proj, mesh, params, spin_mats)
    end

    # Golden Section Search
    ε = 1e-6
    imax = 1000
    resphi = (3.0 - sqrt(5.0)) / 2.0

    τ1 = a + resphi * (b - a)
    τ2 = b - resphi * (b - a)

    f_τ1 = f(τ1)
    f_τ2 = f(τ2)

    k = 0
    while (b - a > ε) && (k < imax)
        if f_τ1 < f_τ2
            b = τ2
            τ2 = τ1
            f_τ2 = f_τ1
            τ1 = a + resphi * (b - a)
            f_τ1 = f(τ1)
        else
            a = τ1
            τ1 = τ2
            f_τ1 = f_τ2
            τ2 = b - resphi * (b - a)
            f_τ2 = f(τ2)
        end
        k += 1
    end

    if f_τ1 < f_τ2
        τ_opt = τ1
        f_opt = f_τ1
    else
        τ_opt = τ2
        f_opt = f_τ2
    end

   get_retraction!(u_proj, u, dn, τ_opt, M, C_mag, retraction)

    # Deep copy to avoid aliasing
    u_out = SpinorState(copy(u_proj.u1), copy(u_proj.u0), copy(u_proj.um1))
    return τ_opt, f_opt, u_out
end

function integrate_linear_contributions(u, d, mesh, params)
    u1, u0, um1 = u.u1, u.u0, u.um1
    d1, d0, dm1 = d.u1, d.u0, d.um1

    V_fun = params.V_fun
    order_finite_element = params.order_finite_element
    q_param = params.q
    p_param = params.p

    x_mesh = mesh.x
    T = mesh.T
    bn = mesh.bn
    global_to_interior = mesh.global_to_interior

    no_of_basis = (order_finite_element == 1) ? 3 : 6

    w = @SVector [0.225, 0.132394152788506, 0.132394152788506, 0.132394152788506, 0.125939180544827, 0.125939180544827, 0.125939180544827]
    α = @SVector [0.059715871789770, 0.797426985353087]
    β = @SVector [0.470142064105115, 0.101286507323456]
    
    x_quad = (
        SVector(1 / 3, 1 / 3),
        SVector(β[1], β[1]),
        SVector(α[1], β[1]),
        SVector(β[1], α[1]),
        SVector(β[2], β[2]),
        SVector(α[2], β[2]),
        SVector(β[2], α[2]),
    )

    no_of_quad_points = length(x_quad)

    D2 = 0.0

    UD_1 = 0.0
    UD_0 = 0.0
    UD_m1 = 0.0

    U2_1 = 0.0
    U2_0 = 0.0
    U2_m1 = 0.0

    @inbounds for k in axes(T, 1)
        tri1 = T[k, 1]
        tri3 = (order_finite_element == 1) ? T[k, 2] : T[k, 3]
        tri5 = (order_finite_element == 1) ? T[k, 3] : T[k, 5]

        z1 = SVector(x_mesh[tri1, 1], x_mesh[tri1, 2])
        z2 = SVector(x_mesh[tri3, 1], x_mesh[tri3, 2])
        z3 = SVector(x_mesh[tri5, 1], x_mesh[tri5, 2])

        J11 = z2[1] - z1[1]
        J12 = z3[1] - z1[1]
        J21 = z2[2] - z1[2]
        J22 = z3[2] - z1[2]

        detBT = 0.5 * (J11 * J22 - J12 * J21)
        invDet = 1.0 / (J11 * J22 - J12 * J21)

        # Inverse transpose Jacobian
        iJ11 =  J22 * invDet
        iJ12 = -J21 * invDet
        iJ21 = -J12 * invDet
        iJ22 =  J11 * invDet

        u1_loc  = MVector{6, ComplexF64}(undef)
        u0_loc  = MVector{6, ComplexF64}(undef)
        um1_loc = MVector{6, ComplexF64}(undef)

        d1_loc  = MVector{6, ComplexF64}(undef)
        d0_loc  = MVector{6, ComplexF64}(undef)
        dm1_loc = MVector{6, ComplexF64}(undef)

        for i in 1:no_of_basis
            node_i = T[k, i]

            if bn[node_i] == 0
                gi = global_to_interior[node_i]

                u1_loc[i]  = ComplexF64(u1[gi])
                u0_loc[i]  = ComplexF64(u0[gi])
                um1_loc[i] = ComplexF64(um1[gi])

                d1_loc[i]  = ComplexF64(d1[gi])
                d0_loc[i]  = ComplexF64(d0[gi])
                dm1_loc[i] = ComplexF64(dm1[gi])
            else
                u1_loc[i]  = 0.0 + 0.0im
                u0_loc[i]  = 0.0 + 0.0im
                um1_loc[i] = 0.0 + 0.0im

                d1_loc[i]  = 0.0 + 0.0im
                d0_loc[i]  = 0.0 + 0.0im
                dm1_loc[i] = 0.0 + 0.0im
            end
        end

        phi_quad = NTuple{7, SVector{6, Float64}}(
        begin
            xi1, xi2 = x_quad[qpt]

            if order_finite_element == 1
                SVector(
                    1.0 - xi1 - xi2,
                    xi1,
                    xi2,
                    0.0, 0.0, 0.0
                )
            else
                SVector(
                    (1.0 - xi1 - xi2) * (1.0 - 2.0 * xi1 - 2.0 * xi2),
                    4.0 * xi1 * (1.0 - xi1 - xi2),
                    xi1 * (2.0 * xi1 - 1.0),
                    4.0 * xi1 * xi2,
                    xi2 * (2.0 * xi2 - 1.0),
                    4.0 * xi2 * (1.0 - xi1 - xi2)
                )
            end
        end for qpt in 1:7
          )

      grad_phi_quad = NTuple{7, SVector{6, SVector{2, Float64}}}(
    begin
        xi1, xi2 = x_quad[qpt]

        if order_finite_element == 1
            z = SVector(0.0, 0.0)

            SVector(
                SVector(-1.0, -1.0),
                SVector( 1.0,  0.0),
                SVector( 0.0,  1.0),
                z, z, z
            )
        else
            SVector(
                SVector(
                    4.0 * xi1 + 4.0 * xi2 - 3.0,
                    4.0 * xi1 + 4.0 * xi2 - 3.0
                ),
                SVector(
                    -4.0 * (2.0 * xi1 + xi2 - 1.0),
                    -4.0 * xi1
                ),
                SVector(4.0 * xi1 - 1.0, 0.0),
                SVector(4.0 * xi2, 4.0 * xi1),
                SVector(0.0, 4.0 * xi2 - 1.0),
                SVector(
                    -4.0 * xi2,
                    -4.0 * (xi1 + 2.0 * xi2 - 1.0)
                )
            )
                end
            end for qpt in 1:7
        )
        

        for qpt in 1:no_of_quad_points
            xi = x_quad[qpt]
            phi = phi_quad[qpt]
            grad_phi = grad_phi_quad[qpt]
            wq = w[qpt]

            x_phy = J11 * xi[1] + J12 * xi[2] + z1[1]
            y_phy = J21 * xi[1] + J22 * xi[2] + z1[2]

            V_val = V_fun(SVector(x_phy, y_phy))

            uq1  = 0.0 + 0.0im
            uq0  = 0.0 + 0.0im
            uqm1 = 0.0 + 0.0im

            dq1  = 0.0 + 0.0im
            dq0  = 0.0 + 0.0im
            dqm1 = 0.0 + 0.0im

            g_u1  = SVector(0.0 + 0.0im, 0.0 + 0.0im)
            g_u0  = SVector(0.0 + 0.0im, 0.0 + 0.0im)
            g_um1 = SVector(0.0 + 0.0im, 0.0 + 0.0im)

            g_d1  = SVector(0.0 + 0.0im, 0.0 + 0.0im)
            g_d0  = SVector(0.0 + 0.0im, 0.0 + 0.0im)
            g_dm1 = SVector(0.0 + 0.0im, 0.0 + 0.0im)

            for i in 1:no_of_basis
                phi_i = phi[i]

                uq1  += u1_loc[i]  * phi_i
                uq0  += u0_loc[i]  * phi_i
                uqm1 += um1_loc[i] * phi_i

                dq1  += d1_loc[i]  * phi_i
                dq0  += d0_loc[i]  * phi_i
                dqm1 += dm1_loc[i] * phi_i

                g_phys = SVector(
                    iJ11 * grad_phi[i][1] + iJ12 * grad_phi[i][2],
                    iJ21 * grad_phi[i][1] + iJ22 * grad_phi[i][2]
                )

                g_u1  += u1_loc[i]  * g_phys
                g_u0  += u0_loc[i]  * g_phys
                g_um1 += um1_loc[i] * g_phys

                g_d1  += d1_loc[i]  * g_phys
                g_d0  += d0_loc[i]  * g_phys
                g_dm1 += dm1_loc[i] * g_phys
            end
            grad_d_sq = real(
                dot(g_d1, g_d1) +
                dot(g_d0, g_d0) +
                dot(g_dm1, g_dm1)
            )

            grad_ud_1  = real(dot(g_u1, g_d1))
            grad_ud_0  = real(dot(g_u0, g_d0))
            grad_ud_m1 = real(dot(g_um1, g_dm1))

            grad_u_sq_1  = real(dot(g_u1, g_u1))
            grad_u_sq_0  = real(dot(g_u0, g_u0))
            grad_u_sq_m1 = real(dot(g_um1, g_um1))

            mass_d_1  = abs2(dq1)
            mass_d_0  = abs2(dq0)
            mass_d_m1 = abs2(dqm1)

            mass_ud_1  = real(uq1  * conj(dq1))
            mass_ud_0  = real(uq0  * conj(dq0))
            mass_ud_m1 = real(uqm1 * conj(dqm1))

            mass_u_1  = abs2(uq1)
            mass_u_0  = abs2(uq0)
            mass_u_m1 = abs2(uqm1)

            fac = wq * detBT

            D2 += fac * (
                0.5*grad_d_sq +
                (V_val + q_param - p_param) * mass_d_1 +
                V_val * mass_d_0 +
                (V_val + q_param + p_param) * mass_d_m1
            )

            UD_1 += fac * (
                0.5*grad_ud_1 +
                (V_val + q_param - p_param) *
                mass_ud_1
            )

            UD_0 += fac * (
                0.5*grad_ud_0 +
                V_val * mass_ud_0
            )

            UD_m1 += fac * (
                0.5*grad_ud_m1 +
                (V_val + q_param + p_param) *
                mass_ud_m1
            )

            # U² contributions
            U2_1 += fac * (
                 0.5*grad_u_sq_1 +
                ( V_val + q_param - p_param) *
                mass_u_1
            )

            U2_0 += fac * (
                0.5*grad_u_sq_0 +
                V_val * mass_u_0
            )

            U2_m1 += fac * (
                0.5*grad_u_sq_m1 +
                (V_val +  q_param + p_param) *
                mass_u_m1
            )
        end
    end

    #E(u + τd - au - bLu) = E(γu + τd), γ = [1-a-b, 1-a, 1-a+b]
    # = τ^2 ∫ (∑ 0.5 |∇d_l|^2 + (V(x) + ql^2 - pl)|d_l|^2) 
    # + 2τ ∫ ∑ 0.5 γ_l Re(∇u_l⋅conj(∇d_l)) + (V(x) + ql^2 - pl)γ_l Re(u_l⋅conj(d_l))
    # + ∫ ∑ 0.5⋅γ_l^2|∇u_l|^2 + (V(x) + ql^2-pl)γ_l^2|∇u_l|^2
    # = τ^2 ⋅ D2 + 2τ(γ1⋅UD_1 + γ0⋅UD_0 + γm1⋅UD_m1) + (γ1^2⋅U2_1 + γ0^2⋅U2_0 + γm1^2⋅U2_m1)
    return D2, UD_1, UD_0, UD_m1, U2_1, U2_0, U2_m1
end

struct DensityPairCoeffs
    D4::Float64
    D3a::Float64
    D3b::Float64
    D2a::Float64
    D2b::Float64
    D2c::Float64
    D1a::Float64
    D1b::Float64
    D0::Float64
end

function precompute_density_pair(ui, di, uj, dj, mesh, params)
    order_finite_element = params.order_finite_element
    x = mesh.x
    T = mesh.T 
    bn = mesh.bn
    g2i = mesh.global_to_interior

    no_of_basis = (order_finite_element == 1) ? 3 : 6

    w = @SVector [0.225, 0.132394152788506, 0.132394152788506, 0.132394152788506, 0.125939180544827, 0.125939180544827, 0.125939180544827]
    α = @SVector [0.059715871789770, 0.797426985353087]
    β = @SVector [0.470142064105115, 0.101286507323456]

    x_quad = (
        SVector(1/3, 1/3),
        SVector(β[1], β[1]),
        SVector(α[1], β[1]), 
        SVector(β[1], α[1]),
        SVector(β[2], β[2]), 
        SVector(α[2], β[2]), 
        SVector(β[2], α[2])
    )


    phi_quad = NTuple{7, SVector{6, Float64}}(
        begin
            xi1, xi2 = x_quad[qpt]
            if order_finite_element == 1
                SVector(-xi1 - xi2 + 1.0, xi1, xi2, 0.0, 0.0, 0.0)
            else
                SVector(
                    (1 - xi1 - xi2) * (1 - 2*xi1 - 2*xi2),
                    4 * xi1 * (1 - xi1 - xi2),
                    xi1 * (2*xi1 - 1),
                    4 * xi1 * xi2,
                    xi2 * (2*xi2 - 1),
                    4 * xi2 * (1 - xi1 - xi2)
                )
            end
        end for qpt in 1:7
    )

    D4 = 0.0
    D3a, D3b = 0.0, 0.0
    D2a, D2b, D2c = 0.0, 0.0, 0.0
    D1a, D1b = 0.0, 0.0
    D0 = 0.0

    @inbounds for k in axes(T, 1)
        tri1, tri2, tri3 = T[k, 1], T[k, 2], T[k, 3]
        z1 = SVector(x[tri1, 1], x[tri1, 2])
        
        if order_finite_element == 1
            z2 = SVector(x[tri2, 1], x[tri2, 2])
            z3 = SVector(x[tri3, 1], x[tri3, 2])
        else
            z2 = SVector(x[T[k, 3], 1], x[T[k, 3], 2]) 
            z3 = SVector(x[T[k, 5], 1], x[T[k, 5], 2]) 
        end

        # Jacobian determinant
        detBT = 0.5 * (z2[1] - z1[1]) * (z3[2] - z1[2]) - (z3[1] - z1[1]) * (z2[2] - z1[2])

        ui_loc = MVector{6, ComplexF64}(undef)
        uj_loc = MVector{6, ComplexF64}(undef)
        di_loc = MVector{6, ComplexF64}(undef)
        dj_loc = MVector{6, ComplexF64}(undef)

        for i in 1:no_of_basis
            node = T[k, i]
            if bn[node] == 0
                gi = g2i[node]
                ui_loc[i]  = ComplexF64(ui[gi])
                di_loc[i]  = ComplexF64(di[gi])
                uj_loc[i]  = ComplexF64(uj[gi])
                dj_loc[i]  = ComplexF64(dj[gi])
            else
                ui_loc[i]  = 0.0im; di_loc[i]  = 0.0im
                uj_loc[i]  = 0.0im; dj_loc[i]  = 0.0im
            end
        end
        for qpt in 1:7
            wq  = w[qpt]
            phi = phi_quad[qpt]

            uqi = sum(phi[i] * ui_loc[i] for i in 1:no_of_basis)
            uqj = sum(phi[i] * uj_loc[i] for i in 1:no_of_basis)
            dqi = sum(phi[i] * di_loc[i] for i in 1:no_of_basis)
            dqj = sum(phi[i] * dj_loc[i] for i in 1:no_of_basis)
            
            abs_ui_sq = abs2(uqi)
            abs_uj_sq = abs2(uqj)
            abs_di_sq = abs2(dqi)
            abs_dj_sq = abs2(dqj)

            re_uqi_dqi = real(uqi * conj(dqi))
            re_uqj_dqj = real(uqj * conj(dqj))

            fD4  = abs_di_sq * abs_dj_sq
            fD3a = 2.0 * abs_di_sq * re_uqj_dqj
            fD3b = 2.0 * abs_dj_sq * re_uqi_dqi
            fD2a = 4.0 * re_uqi_dqi * re_uqj_dqj
            fD2b = abs_ui_sq * abs_dj_sq
            fD2c = abs_uj_sq * abs_di_sq
            fD1a = 2.0 * abs_ui_sq * re_uqj_dqj
            fD1b = 2.0 * abs_uj_sq * re_uqi_dqi
            fD0  = abs_ui_sq * abs_uj_sq

            fac = wq * detBT

            D4  += fac * fD4
            D3a += fac * fD3a
            D3b += fac * fD3b
            D2a += fac * fD2a
            D2b += fac * fD2b
            D2c += fac * fD2c
            D1a += fac * fD1a
            D1b += fac * fD1b
            D0  += fac * fD0
        end
    end

    return DensityPairCoeffs(D4, D3a, D3b, D2a, D2b, D2c, D1a, D1b, D0)
end

function coeff_density_pair(t, γ,  i, j, coeffs::DensityPairCoeffs)
    γi = γ[i]
    γj = γ[j]

    b4 = coeffs.D4 
    b3 = γj * coeffs.D3a + γi * coeffs.D3b 
    b2 = γi * γj * coeffs.D2a + γi^2 * coeffs.D2b + γj^2 * coeffs.D2c 
    b1 = γi^2 * γj * coeffs.D1a + γi * γj^2 * coeffs.D1b 
    b0 = γi^2 * γj^2 * coeffs.D0 

    return b4 * t^4 + b3 * t^3 + b2 * t^2 + b1 * t + b0
end

struct SpinCoeffs
    s4::Float64
    s3_m1::Float64; s3_0::Float64; s3_1::Float64
    s2_m1_1::Float64; s2_m1_0::Float64; s2_0_1::Float64; s2_00::Float64
    s1_m1_00::Float64; s1_m1_0_1::Float64; s1_00_1::Float64
    s0_m1_00_1::Float64
end

function compute_spin_exchange_coeffs(
    u::SpinorState, 
    d::SpinorState, 
    mesh, 
    params,
)
    u1, u0, um1 = u.u1, u.u0, u.um1
    d1, d0, dm1 = d.u1, d.u0, d.um1
    order_finite_element = params.order_finite_element
    x = mesh.x
    T = mesh.T 
    bn = mesh.bn
    g2i = mesh.global_to_interior

    if order_finite_element == 1
        no_of_basis = 3
    elseif order_finite_element == 2
        no_of_basis = 6
    end

    w = @SVector [0.225, 0.132394152788506, 0.132394152788506, 0.132394152788506, 0.125939180544827, 0.125939180544827, 0.125939180544827]
    α = @SVector [0.059715871789770, 0.797426985353087]
    β = @SVector [0.470142064105115, 0.101286507323456]

    x_quad = (
        SVector(1/3, 1/3),
        SVector(β[1], β[1]),
        SVector(α[1], β[1]), 
        SVector(β[1], α[1]),
        SVector(β[2], β[2]), 
        SVector(α[2], β[2]), 
        SVector(β[2], α[2])
    )

    s4 = 0.0 #∫ 2d_{-1}d_1\conj{d_0}^2
    s3_m1, s3_0, s3_1 = 0.0, 0.0, 0.0 #∫ 2u_{-1}d_1\conj{d_0}^2, ∫2d_{-1}d_1\conj{u_0d_0}, ∫2u_1d_{-1}\conj{d_0}^2
    s2_m1_1, s2_m1_0, s2_0_1, s2_00 = 0.0, 0.0, 0.0, 0.0 # ∫2u_{-1}u1\conj{d_0}^2, ∫2u_{-1}d1\conj{u_0}d_0, ∫2u_1 d_{-1}\conj{u_0d_0}, ∫2d_{-1}d1\conj{u_0}^2
    s1_m1_00, s1_m1_0_1, s1_00_1 = 0.0, 0.0, 0.0 # ∫2u_{-1}d1\conj{u_0}^2, ∫2u_{m1}u_1\conj{u0d0}, ∫ 2u1d_{-1}\conj{u_0}^2
    s0_m1_00_1 = 0.0 #∫2u_{-1}u1\conj{u_0}^2


    phi_quad = NTuple{7, SVector{6, Float64}}(
    begin
        xi1, xi2 = x_quad[qpt]

        if order_finite_element == 1
            SVector(
                1.0 - xi1 - xi2,
                xi1,
                xi2,
                0.0, 0.0, 0.0
            )
        else
            SVector(
                (1.0 - xi1 - xi2) * (1.0 - 2.0 * xi1 - 2.0 * xi2),
                4.0 * xi1 * (1.0 - xi1 - xi2),
                xi1 * (2.0 * xi1 - 1.0),
                4.0 * xi1 * xi2,
                xi2 * (2.0 * xi2 - 1.0),
                4.0 * xi2 * (1.0 - xi1 - xi2)
            )
        end
    end for qpt in 1:7
        )

    @inbounds for k in axes(T, 1)
        tri1, tri2, tri3 = T[k, 1], T[k, 2], T[k, 3]
        z1 = SVector(x[tri1, 1], x[tri1, 2])
        
        if order_finite_element == 1
            z2 = SVector(x[tri2, 1], x[tri2, 2])
            z3 = SVector(x[tri3, 1], x[tri3, 2])
        else
            z2 = SVector(x[T[k, 3], 1], x[T[k, 3], 2]) 
            z3 = SVector(x[T[k, 5], 1], x[T[k, 5], 2]) 
        end

        detBT = 0.5*((z2[1] - z1[1]) * (z3[2] - z1[2]) - (z3[1] - z1[1]) * (z2[2] - z1[2]))
        u1_loc  = MVector{6, ComplexF64}(undef); d1_loc  = MVector{6, ComplexF64}(undef)
        u0_loc  = MVector{6, ComplexF64}(undef); d0_loc  = MVector{6, ComplexF64}(undef)
        um1_loc = MVector{6, ComplexF64}(undef); dm1_loc = MVector{6, ComplexF64}(undef)

        for i in 1:no_of_basis
            node = T[k, i]
            if bn[node] == 0
                gi = g2i[node]
                u1_loc[i]  = ComplexF64(u1[gi])
                d1_loc[i]  = ComplexF64(d1[gi])
                u0_loc[i]  = ComplexF64(u0[gi])
                d0_loc[i]  = ComplexF64(d0[gi])
                um1_loc[i] = ComplexF64(um1[gi])
                dm1_loc[i] = ComplexF64(dm1[gi])
            else
                u1_loc[i]  = 0.0im; d1_loc[i]  = 0.0im
                u0_loc[i]  = 0.0im; d0_loc[i]  = 0.0im
                um1_loc[i] = 0.0im; dm1_loc[i] = 0.0im
            end
        end

        for qpt in 1:7
            wq  = w[qpt]
            phi = phi_quad[qpt]
            
            u1_q = 0.0im
            u0_q = 0.0im
            um1_q = 0.0im

            d1_q = 0.0im
            d0_q = 0.0im
            dm1_q = 0.0im

            @inbounds for i in 1:no_of_basis
                phi_i = phi[i]

                u1_q  += phi_i * u1_loc[i]
                u0_q  += phi_i * u0_loc[i]
                um1_q += phi_i * um1_loc[i]

                d1_q  += phi_i * d1_loc[i]
                d0_q  += phi_i * d0_loc[i]
                dm1_q += phi_i * dm1_loc[i]
            end

            # Conjugates
            cu0  = conj(u0_q)
            cd0  = conj(d0_q)

            fac = wq * detBT

            # τ⁴
            s4 += fac * 2.0 * real(d1_q * dm1_q * cd0^2)

            # τ³
            s3_m1 += fac * 2.0 * real(um1_q * d1_q * cd0^2)
            s3_0 += fac * 2.0 * real(d1_q * dm1_q * cu0 * cd0)
            s3_1 += fac * 2.0 * real(u1_q * dm1_q * cd0^2 )

            # τ²
            s2_m1_1 += fac * 2.0 * real(um1_q * u1_q * cd0^2)
            s2_m1_0 += fac * 2.0 * real(um1_q * d1_q * cu0 * cd0)
            s2_0_1 += fac * 2.0 * real(u1_q * dm1_q * cu0 * cd0)
            s2_00 += fac * 2.0 * real( d1_q * dm1_q * cu0^2)

            # τ
            s1_m1_00 += fac * 2.0 * real(um1_q * d1_q * cu0^2)
            s1_m1_0_1 += fac * 2.0 * real(um1_q * u1_q * cu0 * cd0)
            s1_00_1 += fac * 2.0 * real(u1_q * dm1_q * cu0^2)

            s0_m1_00_1 += fac * 2.0 * real( um1_q * u1_q * cu0^2)
        end
    end

    return SpinCoeffs( s4,
        s3_m1, s3_0, s3_1,
        s2_m1_1, s2_m1_0, s2_0_1, s2_00,
        s1_m1_00, s1_m1_0_1, s1_00_1,
        s0_m1_00_1)
end

function golden_section_search(a, b, f)
     ε = 1e-6
    imax = 1000
    resphi = (3.0 - sqrt(5.0)) / 2.0

    τ1 = a + resphi * (b - a)
    τ2 = b - resphi * (b - a)

    f_τ1 = f(τ1)
    f_τ2 = f(τ2)

    k = 0
    while (b - a > ε) && (k < imax)
        if f_τ1 < f_τ2
            b = τ2
            τ2 = τ1
            f_τ2 = f_τ1
            τ1 = a + resphi * (b - a)
            f_τ1 = f(τ1)
        else
            a = τ1
            τ1 = τ2
            f_τ1 = f_τ2
            τ2 = b - resphi * (b - a)
            f_τ2 = f(τ2)
        end
        k += 1
    end


    if f_τ1 < f_τ2
        return τ1, f_τ1
    else
        return τ2, f_τ2
    end
end

#L2 normal retraction and optimal step size calculation in one step
function optimal_tau_newton(u, d, mesh, params, M_final, spin_mats, lower_bound, upper_bound)
    C_mag = params.C_mag
    β0 = params.β0 
    β1 = params.β1

    u1  = u.u1 
    u0  = u.u0 
    um1 = u.um1 

    d1  = d.u1 
    d0  = d.u0
    dm1 = d.um1

    D2, UD_1, UD_0, UD_m1, U2_1, U2_0, U2_m1 = integrate_linear_contributions(u, d, mesh, params)

    coeffs11    = precompute_density_pair(u1,  d1,  u1,  d1,  mesh, params) #∫|γ1⋅u_1 + τ⋅d_1|^4
    coeffs00    = precompute_density_pair(u0,  d0,  u0,  d0,  mesh, params)#∫|γ0⋅u0  τ⋅d_0|^4
    coeffs_m1_m1= precompute_density_pair(um1, dm1, um1, dm1, mesh, params)#∫|γm1⋅um1 +  τ⋅d_m1|^4

    coeffs_01   = precompute_density_pair(u1,  d1,  u0,  d0,  mesh, params) # u1-u0 pair
    coeffs_um1_1= precompute_density_pair(u1,  d1,  um1, dm1, mesh, params) # u1-um1 pair
    coeffs_um1_0= precompute_density_pair(u0,  d0,  um1, dm1, mesh, params) # u0-um1 pair

    coeffs_spin = compute_spin_exchange_coeffs(u, d, mesh, params)

    N_dm1 = real(dot(dm1, M_final, dm1))
    N_d0  = real(dot(d0,  M_final, d0))
    N_d1  = real(dot(d1,  M_final, d1))
    d_norm = N_dm1 + N_d0 + N_d1

    L2_LuLd = real(dot(um1, M_final, dm1) + dot(u1, M_final, d1))
    LuLu     = real(dot(um1, M_final, um1) + dot(u1, M_final, u1))
    Ld_M_d =  real(dot(d1, M_final, d1) - dot(dm1, M_final, dm1))
    dot_u_d = real(dot(um1, M_final, dm1) + dot(u0, M_final, d0) + dot(u1, M_final, d1)) #should be 0
    dot_Lu_d = real(dot(u1, M_final, d1) - dot(um1, M_final, dm1)) #should be 0

    if LuLu-C_mag^2 < -1e-12 * max(abs(LuLu), abs(C_mag^2), 1.0)
            @show(LuLu-C_mag^2)
            error("Regularity check failed")
    end
    
 
    function eval_trial_state(t)
        α = 0.0
        β = 0.0
        rhs_N = 10.0 
        rhs_M = 10.0

        u_out_sq = 1.0 + 2*t*dot_u_d + t^2 * d_norm #(u + td, u + td) = 1 + 2t(u,d) + t^2(d,d) = 1 + t^2*d_nprm
        dot_Lu_out_Lu = LuLu + t * L2_LuLd # (Lu + tLd, Lu)= (Lu,Lu) + t(Lu, Ld)
        dot_Lu_out_u_out = C_mag  + 2 * t * dot_Lu_d + t^2 * Ld_M_d # (Lu + tLd, u + td) = (Lu, u) + t(Lu, d) + t(u, Ld) + t^2(Ld,d) = C_mag + t^2(Ld,d)
        dot_u_out_u = 1.0 + t * dot_u_d #should be 1
        dot_u_out_Lu = C_mag + t * dot_Lu_d #should be C_mag

        count = 1
        while (abs(rhs_N) > 1e-12 || abs(rhs_M) > 1e-12) && count <= 5000
            rhs_N = α^2 + β^2 * LuLu -  2.0 * α * dot_u_out_u - 2.0 * β * dot_u_out_Lu + 2.0 * α * β * C_mag + u_out_sq - 1.0
            rhs_M = α^2 * C_mag + β^2 * C_mag - 2.0 * α * dot_u_out_Lu - 2.0 * β * dot_Lu_out_Lu + 2.0 * α * β * LuLu + dot_Lu_out_u_out - C_mag

            partial_N_α = 2.0 * α + 2.0 * β * C_mag - 2.0 * dot_u_out_u
            partial_N_β = 2.0 * β * LuLu + 2.0 * α * C_mag - 2.0 * dot_u_out_Lu
            partial_M_α = 2.0 * α * C_mag + 2.0 * β * LuLu - 2.0 * dot_u_out_Lu
            partial_M_β = 2.0 * β * C_mag + 2.0 * α * LuLu - 2.0 * dot_Lu_out_Lu

            detJ = (partial_N_α * partial_M_β - partial_N_β * partial_M_α)
            det_inv = 1.0 / detJ

            α -= det_inv * (partial_M_β * rhs_N - partial_N_β * rhs_M)
            β -= det_inv * (partial_N_α * rhs_M - partial_M_α * rhs_N)

            count += 1       
        end

        γ1  = 1.0 - α - β
        γ0  = 1.0 - α
        γm1 = 1.0 - α + β
        γ = SVector(γ1, γ0, γm1) 

        l2 = D2 
        l1 = γ1 * UD_1 + γ0 * UD_0 + γm1 * UD_m1 
        l0 = γ1^2 * U2_1 + γ0^2 * U2_0 + γm1^2 * U2_m1 
        linear_part = l2 * t^2 + 2 * l1 * t + l0
        
        u1_quartic  = coeff_density_pair(t, γ, 1, 1, coeffs11)
        u0_quartic  = coeff_density_pair(t, γ, 2, 2, coeffs00)
        um1_quartic = coeff_density_pair(t, γ, 3, 3, coeffs_m1_m1)

        u1_u0_sq  = coeff_density_pair(t, γ, 1, 2, coeffs_01)
        u1_um1_sq = coeff_density_pair(t, γ, 1, 3, coeffs_um1_1)
        u0_um1_sq = coeff_density_pair(t, γ, 2, 3, coeffs_um1_0)

        density_term = u1_quartic + u0_quartic + um1_quartic + 2 * (u1_u0_sq + u1_um1_sq + u0_um1_sq)

        spin_density_term = u1_quartic - 2 * u1_um1_sq + um1_quartic + 2 * (u1_u0_sq + u0_um1_sq)
    
        c4 = coeffs_spin.s4

        c3 = γm1 * coeffs_spin.s3_m1 +
            2 * γ0 * coeffs_spin.s3_0 +
            γ1 * coeffs_spin.s3_1

        c2 = γ1 * γm1 * coeffs_spin.s2_m1_1 +
            2 * γ0 * γm1 * coeffs_spin.s2_m1_0 +
            2 * γ0 * γ1 * coeffs_spin.s2_0_1 +
            γ0^2 * coeffs_spin.s2_00

        c1 = γ0^2 * γm1 * coeffs_spin.s1_m1_00 +
            2 * γ0 * γ1 * γm1 * coeffs_spin.s1_m1_0_1 +
            γ0^2 * γ1 * coeffs_spin.s1_00_1

        c0 = γ0^2 * γ1 * γm1 * coeffs_spin.s0_m1_00_1

        spin_term = 2*(c4 * t^4 + c3 * t^3 + c2 * t^2 + c1 * t + c0) + spin_density_term

        return real(linear_part + 0.5*β0 * density_term + 0.5*β1 * spin_term), α, β
    end

    τ, E_new = golden_section_search(lower_bound, upper_bound, t -> first(eval_trial_state(t)))

    _, α_opt, β_opt = eval_trial_state(τ)

    u1_new  = @. u1  + τ * d1  - (α_opt + β_opt) * u1 
    u0_new  = @. u0  + τ * d0  - α_opt * u0 
    um1_new = @. um1 + τ * dm1 - (α_opt - β_opt) * um1 
   
    u_new = SpinorState(u1_new, u0_new, um1_new)
    return τ, E_new, u_new
end

function get_new_energy(u::SpinorState, u_old::SpinorState, d::SpinorState, d_old::SpinorState, mesh, params, M, spin_mats, lower_bound::Float64, upper_bound::Float64, τ_fixed::Float64, step_control::String, retraction::String, ws::Workspace, metric)
    u_proj = ws.u_proj
    if step_control == "optimal"
        if retraction == "newton"
            τ, E_new, u_new = optimal_tau_newton(u, d, mesh, params, M, spin_mats, lower_bound, upper_bound)
        else
            τ, E_new, u_new = optimal_tau_blackbox(u, d, u_proj, mesh, params, M, spin_mats, lower_bound, upper_bound, retraction)
        end
    elseif step_control == "fixed"
        τ = τ_fixed
        get_retraction!(u_proj, u, d, τ, M, params.C_mag, retraction)
        E_new = spin1_energy(u_proj, mesh, params, spin_mats)
        u_new = u_proj
    elseif step_control == "MM"

       du_sq = sum(abs2, u.u1 .- u_old.u1) + 
                sum(abs2, u.u0 .- u_old.u0) + 
                sum(abs2, u.um1 .- u_old.um1)

        dg_sq = sum(abs2, d.u1 .- d_old.u1) + 
                sum(abs2, d.u0 .- d_old.u0) + 
                sum(abs2, d.um1 .- d_old.um1)

        τ_prelim = sqrt(du_sq / dg_sq)
        τ = min(max(τ_prelim, lower_bound), upper_bound)
        get_retraction!(u_proj, u, d, τ, M, params.C_mag, retraction)

        E_new = spin1_energy(u_proj, mesh, params, spin_mats)
        u_new = SpinorState(copy(u_proj.u1), copy(u_proj.u0), copy(u_proj.um1))
    elseif step_control == "BB"#BB long
        du1 = u.u1 - u_old.u1 
        du0 = u.u0 - u_old.u0 
        dum1 = u.um1 - u_old.um1 

        du_vec = [real(du1);imag(du1);real(du0);imag(du0);real(dum1);imag(dum1)]
        du_vec = [du1; du0; dum1]

        dg1 = d.u1 - d_old.u1 
        dg0 = d.u0 - d_old.u0 
        dgm1 = d.um1 - d_old.um1 
        dg_vec = [dg1; dg0; dgm1]

        w = @SVector [0.225, 0.132394152788506, 0.132394152788506, 0.132394152788506, 0.125939180544827, 0.125939180544827, 0.125939180544827]
        α = @SVector [0.059715871789770, 0.797426985353087]
        β = @SVector [0.470142064105115, 0.101286507323456]
        
        x_quad = (
            SVector(1/3, 1/3),
            SVector(β[1], β[1]), 
            SVector(α[1], β[1]), 
            SVector(β[1], α[1]),  
            SVector(β[2], β[2]), 
            SVector(α[2], β[2]), 
            SVector(β[2], α[2])
        )
        no_of_quad_points = 7

        order = params.order_finite_element
        no_of_basis = (order == 1) ? 3 : 6

        phi_in_quad_point = MMatrix{6, 7, Float64}(undef)
        for qpt in 1:no_of_quad_points
            xq1, xq2 = x_quad[qpt][1], x_quad[qpt][2]
            if order == 1
                phi_in_quad_point[1, qpt] = -xq1 - xq2 + 1.0
                phi_in_quad_point[2, qpt] = xq1
                phi_in_quad_point[3, qpt] = xq2
            else
                phi_in_quad_point[1, qpt] = (1 - xq1 - xq2) * (1 - 2*xq1 - 2*xq2)
                phi_in_quad_point[2, qpt] = 4 * xq1 * (1 - xq1 - xq2)
                phi_in_quad_point[3, qpt] = xq1 * (2*xq1 - 1)
                phi_in_quad_point[4, qpt] = 4 * xq1 * xq2
                phi_in_quad_point[5, qpt] = xq2 * (2*xq2 - 1)
                phi_in_quad_point[6, qpt] = 4 * xq2 * (1 - xq1 - xq2)
            end
        end

        Adu_out = similar(du_vec)
        Adg_out = similar(dg_vec)

        apply_A_parallel!(
            Adu_out, du_vec,
            u.u1, u.u0, u.um1,
            mesh, params, length(u.u1), no_of_basis,
            w, phi_in_quad_point, params.β0, params.β1,
            ws, metric
        )

        apply_A_parallel!(
            Adg_out, dg_vec,
            u.u1, u.u0, u.um1,
            mesh, params, length(u.u1), no_of_basis,
            w, phi_in_quad_point, params.β0, params.β1,
            ws, metric
        )

        τ_prelim = -real(dot(du_vec, Adu_out))/real(dot(du_vec, Adg_out))
       
         τ = min(max(τ_prelim, lower_bound), upper_bound)
        #   @show(τ_prelim, τ)
        get_retraction!(u_proj, u, d, τ, M, params.C_mag, retraction)

        E_new = spin1_energy(u_proj, mesh, params, spin_mats)
        u_new = SpinorState(copy(u_proj.u1), copy(u_proj.u0), copy(u_proj.um1))
    end

    return τ, E_new, u_new
end

function spin1_gradient_flow(
    u::SpinorState, mesh, params, M;
    metric::String = "spin",
    step_control::String = "optimal",
    retraction::String = "newton",
    fixed_τ::Float64 = 1.0
)
    @assert metric in ["spin", "spin-diagonal", "density", "A0"] "Invalid metric: $metric"
    @assert step_control in ["optimal", "fixed", "MM", "BB"] "Invalid step_control: $step_control"
    @assert retraction in ["newton", "exact", "inexact", "closed-form"] "Invalid retraction: $retraction"

    order_finite_element = params.order_finite_element
    Energies = Float64[]
    Residuals = Float64[]
    time_vec = Float64[]
    Diffs = Float64[]
    N1_vec = Float64[]
    N0_vec = Float64[]
    Nm1_vec = Float64[]
    taus = Float64[]

    spin_mats = SpinMatrices(params.F)
    E = spin1_energy(u, mesh, params, spin_mats)

    @printf("Initial energy: %.12f | Metric: %s | Step: %s | Retraction: %s\n", E, metric, step_control, retraction)
    push!(Energies, E)

    μ, λ = 0.0, 0.0
    res = 10.0
    dn_norm = 0.0
    C_mag = params.C_mag  
    q = params.q
    p = params.p    
    # buffers
    buf_m1 = similar(u.um1)
    buf_0  = similar(u.u0)
    buf_1  = similar(u.u1)
 
    dn = SpinorState(similar(u.u1), similar(u.u0), similar(u.um1))
    u_old      = SpinorState(similar(u.u1), similar(u.u0), similar(u.um1))
    dn_old = SpinorState(similar(u.u1), similar(u.u0), similar(u.um1))


    ws = Workspace(u, size(mesh.T,1))
    M, S, V = matrices(mesh, params)

    # Build preconditioner for CG 
        P = lu(2 * blockdiag(0.5 * S+ V + (q - p) * M, 
                    0.5 * S + V, 
                    0.5 * S+ V + (q + p) * M))

    τ = fixed_τ
    lower_bound = 1e-4
    if step_control == "BB" || step_control == "optimal"
        upper_bound = 20.0
        if metric == "A0"
            upper_bound = 4.0
        end
    else 
        upper_bound = 10000.0
    end

    energy_diminishing = true

    for j = 1:params.maxIter
        t_start = time()

        E_prime = frechet_derivative!(u, mesh, params, ws) #returns SpinorState
        res, μ, λ, Mu = get_residual!(u, params, M, E_prime, ws)
        push!(Residuals, res)

         gradR, μ_A, λ_A = get_Riemannian_gradient(u, Mu, E_prime, mesh, params, metric, P, res, ws)

        if j > 1
            copyto!(dn_old.u1,  dn.u1)
            copyto!(dn_old.u0,  dn.u0)
            copyto!(dn_old.um1, dn.um1)
        end

        dn.u1  .= -gradR.u1
        dn.u0  .= -gradR.u0
        dn.um1 .= -gradR.um1

        dn = SpinorState(copy(dn.u1), copy(dn.u0), copy(dn.um1))
    
        mul!(buf_m1, M, dn.um1)
        mul!(buf_0,  M, dn.u0)
        mul!(buf_1,  M, dn.u1)

        tangent_check_N = real(dot(u.um1, buf_m1) + dot(u.u0, buf_0) + dot(u.u1, buf_1))
        tangent_check_M = real(-dot(u.um1, buf_m1) + dot(u.u1, buf_1))

        if abs(tangent_check_N) > 1e-8 || abs(tangent_check_M) > 1e-8
            @show(tangent_check_M, tangent_check_N)
        end
        if j <= 2
            if step_control == "MM" || step_control == "BB" 
                 τ, E_new, u_new = get_new_energy(u, u_old, dn, dn_old, mesh, params, M, spin_mats, 1e-6, 0.5, fixed_τ, "optimal", retraction, ws, metric)
            elseif metric == "A0"
                τ, E_new, u_new = get_new_energy(u, u_old, dn, dn_old,  mesh, params, M, spin_mats, 1e-6, 1e-1, fixed_τ, step_control, retraction, ws, metric)
            else
                τ, E_new, u_new = get_new_energy(u, u_old, dn, dn_old,  mesh, params, M, spin_mats, 1e-6, 1.0, fixed_τ, step_control, retraction, ws, metric)
            end
        else
             τ, E_new, u_new = get_new_energy(u, u_old, dn, dn_old, mesh, params, M, spin_mats, lower_bound, upper_bound, fixed_τ, step_control, retraction, ws, metric)          
        end
        push!(taus, τ)
        push!(Energies, E_new)
        
        mul!(buf_m1, M, u_new.um1)
        mul!(buf_0,  M, u_new.u0)
        mul!(buf_1,  M, u_new.u1)

        m_m1 = real(dot(u_new.um1, buf_m1))
        m_0  = real(dot(u_new.u0,  buf_0))
        m_1  = real(dot(u_new.u1,  buf_1))

        push!(N1_vec, m_1)
        push!(N0_vec, m_0)
        push!(Nm1_vec, m_m1)

        sphere_check_N = m_m1 + m_0 + m_1
        sphere_check_M = m_1 - m_m1
        if retraction != "inexact"
            if abs(sphere_check_N - 1.0) > 1e-7 || abs(sphere_check_M - C_mag) > 1e-7
                @show(sphere_check_N, sphere_check_M)
            end
        end
        if retraction == "inexact"
            @show(sphere_check_N, sphere_check_M)
        end

        diff = Energies[j] - Energies[j+1]
        push!(Diffs, diff)
        if (diff < -1e-10) && (retraction != "inexact") 
            @printf("\n Energy increased by %.4e at iteration %d!\n", -diff, j)
            energy_diminishing = false
        end
        copyto!(u_old.u1, u.u1)
        copyto!(u_old.u0, u.u0)
        copyto!(u_old.um1, u.um1)

        u = SpinorState(copy(u_new.u1), copy(u_new.u0), copy(u_new.um1))

        iter_time = time() - t_start
  
        @printf("\n P%d [%s | %s | %s] Iter: %d | Energy: %.12f | μ: %.12f | λ: %.12f | diff: %.12e | Res: %.12e | τ: %.17e | Time: %.3f s\n", 
                order_finite_element, metric, step_control, retraction, j, E_new, μ/2, λ/2, diff, res, τ, iter_time)
        push!(time_vec, iter_time)

        if res < params.tol         
            println("Converged! \n")
            dn_norm = max(maximum(abs, dn.um1), maximum(abs, dn.u0), maximum(abs, dn.u1))
            break
        end

        if j == params.maxIter && res > params.tol
            println("Maximum number of iterations reached. No convergence.")
            dn_norm = max(maximum(abs, dn.um1), maximum(abs, dn.u0), maximum(abs, dn.u1))
            break
        end
       
    end

    total_time = sum(time_vec)/60
    avg_time   = sum(time_vec)/length(time_vec)

    Nm1 = real(dot(u.um1, M, u.um1))
    N0  = real(dot(u.u0, M, u.u0))
    N1  = real(dot(u.u1, M, u.u1))

    energy_status = energy_diminishing ?
    "The scheme was energy diminishing." :
    "The scheme was not energy diminishing."

    result_summary = @sprintf("""
        __________Final Result: [%s | %s | %s]__________
        P%d | Iter: %d | Energy: %.12f | μ: %.12f | λ: %.12f | Res: %.12e | Norm dn: %.12e | Average time per iteration: %.3f s | Total time: %.3f min
        Component masses: N₋₁ = %.12f | N₀ = %.12f | N₁ = %.12f
        %s
        """,
            metric, step_control, retraction, order_finite_element, length(Energies), Energies[end],
            μ / 2, λ / 2, res, dn_norm, avg_time, total_time,
            Nm1, N0, N1, energy_status
        )

    return u, Energies, Residuals, Diffs, dn_norm, μ, λ, taus, N1_vec, N0_vec, Nm1_vec, time_vec, result_summary
end

function P2_reorder_coordinates(coords, coords_inner, u_approx)
    sorted_indices = sortperm(eachrow(coords), by=x -> (x[2], x[1])) #sort by y, then by x
    new_order = coords[sorted_indices, :]

    sorted_indices_inner = sortperm(eachrow(coords_inner), by=x -> (x[2], x[1]))
    um1 = u_approx.um1 
    u0 = u_approx.u0 
    u1 = u_approx.u1

    um1_reordered = um1[sorted_indices_inner]
    u0_reordered = u0[sorted_indices_inner]
    u1_reordered = u1[sorted_indices_inner]

    u_reordered = SpinorState(u1_reordered, u0_reordered, um1_reordered)
    return  u_reordered
end

function plot_density_2D(u, mesh, params)
    um1 = u.um1 
    u0  = u.u0 
    u1  = u.u1 

    Nd = mesh.x 
    a  = params.a 
    b  = params.b 
    N  = params.N 
    order_finite_element = params.order_finite_element

    if order_finite_element == 1
        l = div(length(um1), 2)
        Nx = Int(sqrt(size(Nd,1)) - 1)

        um1_grid = zeros(ComplexF64, Nx + 1, Nx + 1)
        u0_grid  = zeros(ComplexF64, Nx + 1, Nx + 1)
        u1_grid  = zeros(ComplexF64, Nx + 1, Nx + 1)

        for j = 1:Nx - 1
            um1_grid[j+1, 2:end-1] = um1[(j-1)*(Nx-1) + 1 : (Nx-1)*j]
            u0_grid[j+1, 2:end-1]  = u0[(j-1)*(Nx-1) + 1 : (Nx-1)*j]
            u1_grid[j+1, 2:end-1]  = u1[(j-1)*(Nx-1) + 1 : (Nx-1)*j]
        end

        x_range = collect(range(a, b; length = Nx + 1))
        y_range = collect(range(a, b; length = Nx + 1))

        X = zeros(Float64, Nx+1, Nx+1)
        Y = zeros(Float64, Nx+1, Nx+1)

        for i in 1:Nx+1
            for j in 1:Nx+1
                X[i,j] = x_range[j]
                Y[i,j] = y_range[i]
            end
        end

    elseif order_finite_element == 2
        x_P2 = Nd[:, 1]
        y_P2 = Nd[:, 2]

        x_range = collect(LinRange(minimum(x_P2), maximum(x_P2), 2N + 1))
        y_range = collect(LinRange(minimum(y_P2), maximum(y_P2), 2N + 1))

        X = zeros(Float64, length(y_range), length(x_range))
        Y = zeros(Float64, length(y_range), length(x_range))

        for i in 1:length(y_range)
            for j in 1:length(x_range)
                X[i,j] = x_range[j]
                Y[i,j] = y_range[i]
            end
        end

        Nd_inner_nodes = Nd[.!((Nd[:, 1] .== a) .| (Nd[:, 1] .== b) .| (Nd[:, 2] .== a) .| (Nd[:, 2] .== b)), :]
        u_reordered = P2_reorder_coordinates(Nd, Nd_inner_nodes, u)

        um1 = u_reordered.um1
        u0  = u_reordered.u0
        u1 = u_reordered.u1

        Nx = Int(sqrt(size(Nd, 1)) - 1)
        l  = div(length(um1), 2)

        um1_grid = zeros(ComplexF64, Nx + 1, Nx + 1)
        u0_grid  = zeros(ComplexF64, Nx + 1, Nx + 1)
        u1_grid  = zeros(ComplexF64, Nx + 1, Nx + 1)

        for j = 1:Nx - 1
            um1_grid[j+1, 2:end-1] = um1[(j-1)*(Nx-1) + 1 : (Nx-1)*j]
            u0_grid[j+1, 2:end-1]  = u0[(j-1)*(Nx-1) + 1 : (Nx-1)*j]
            u1_grid[j+1, 2:end-1]  = u1[(j-1)*(Nx-1) + 1 : (Nx-1)*j]
        end
    end


    # Density 
    um1_density = abs2.(um1_grid)
    u0_density  = abs2.(u0_grid)
    u1_density  = abs2.(u1_grid)

    # Colormap and Shared Color Scale 
    cmin = min(
        minimum(um1_density),
        minimum(u0_density),
        minimum(u1_density)
    )

    cmax = max(
        maximum(um1_density),
        maximum(u0_density),
        maximum(u1_density)
    ) * 1.05

    crange_density = (cmin, cmax)
    cmap_density = :turbo

fig = CairoMakie.Figure(size = (1600, 550))

density_scale = x -> x^0.7

ticks_size = 28

ax1 = CairoMakie.Axis(
    fig[1, 1],
    title = "|u₁|²",
    titlesize = 50,
    titlegap = 20,
    xlabel = "x",
    ylabel = "y",
    xticklabelsize = ticks_size,
    yticklabelsize = ticks_size,
    aspect = CairoMakie.DataAspect(),
    limits = (-10, 10, -10, 10)
)

h1 = CairoMakie.heatmap!(
    ax1, x_range, y_range, u1_density;
    colormap = cmap_density,
    colorrange = crange_density,
    colorscale = density_scale
)

ax2 = CairoMakie.Axis(
    fig[1, 2],
    title = "|u₀|²",
    titlesize = 50,
    titlegap = 20,
    xlabel = "x",
    ylabel = "y",
    xticklabelsize = ticks_size,
    yticklabelsize = ticks_size,
    aspect = CairoMakie.DataAspect(),
    limits = (-10, 10, -10, 10)
)

CairoMakie.heatmap!(
    ax2, x_range, y_range, u0_density;
    colormap = cmap_density,
    colorrange = crange_density,
    colorscale = density_scale
)

ax3 = CairoMakie.Axis(
    fig[1, 3],
    title = "|u₋₁|²",
    titlesize = 50,
    titlegap = 20,
    xlabel = "x",
    ylabel = "y",
    xticklabelsize = ticks_size,
    yticklabelsize = ticks_size,
    aspect = CairoMakie.DataAspect(),
    limits = (-10, 10, -10, 10)
)

CairoMakie.heatmap!(
    ax3, x_range, y_range, um1_density;
    colormap = cmap_density,
    colorrange = crange_density,
    colorscale = density_scale
)

# Ticks equally spaced in transformed color-scale coordinates
scaled_ticks = range(
    density_scale(cmin),
    density_scale(cmax);
    length = 6
)

density_ticks = scaled_ticks .^ (1 / 0.7)

CairoMakie.Colorbar(
    fig[1, 4],
    h1;
    # label = "Density",
    width = 18,
    height = 400,
    labelpadding = 15,
    ticks = density_ticks,
    ticklabelsize = ticks_size,
    tickformat = values -> [@sprintf("%.3f", v) for v in values]
)

colgap!(fig.layout, 45)

display(fig)

    return X, Y, um1_density, u0_density, u1_density
end




##########################################################################################################
# MAIN PROGRAM STARTS HERE 
##########################################################################################################

# Choose order_finite_element = 1 for P1 FEM, and order_finite_element = 2 for P2 FEM
    order_finite_element = 2 
# Domain: [a,b]^2
    a = -16.0
    b = 16.0 
# Specify number of subintervals on each axis
    N = 256 
# Nonlinear interaction parameters
    β0 = 240.8
    β1 = 7.5
# Linear Zeeman parameter
    p = 0.0
# Quadratic Zeeman parameter
    q = 0.5
# Spin-F
    F = 1
# Specify potential
V_fun(x) =  0.5*(x[1]^2 + x[2]^2) + 25*sin(pi*x[1]/2)^2 + 25*sin(pi*x[2]/2)^2
# Specify magnetization constant
    C_mag = 0.3 
# Specify maximum number of iterations
    maxIter = 6000
# Stopping criterion: Res < tol
tol = 1e-8

# Initialize mesh and parameter structs
    params = Params(order_finite_element, a, b, F, N, β0, β1, p, q, V_fun, C_mag, maxIter, tol)
    mesh = fem2d_uniform_mesh(params)
# build mass, stiffness and potential matrix ONCE
    M, S, V = matrices(mesh, params)

# Initial value
    ζ(x) = pi^(-1/2)*exp(-0.5*(x[1]^2 + x[2]^2))
    init_phi = initial_phi(mesh,  ζ, M)
    u_init = SpinorState(sqrt(.475) * init_phi, sqrt(0.35) * init_phi, sqrt(0.175) * init_phi)

#Gradient flow 
    # Choose metric: 
        # "A0" (shifted H_0^1 metric), 
        # "density" (density-adaptive metric)
        # "spin-diagonal" (diagonal spin-adaptive metric)
        # "spin" (full spin-adaptive metric)
    # Choose step control:
        # "fixed", τ = 1.0 by default
        # "MM"
        # "BB" 
        # "optimal" 
    # Choose retraction:
        # "newton" (L2-normal retraction)
        # "exact"
        # "inexact"
        # "closed-form"

    (u_final, Energies, Residuals, Diffs, dn_norm, μ, λ, taus, N1_vec, N0_vec, Nm1_vec, time_vec, result_summary) = spin1_gradient_flow(
                u_init, mesh, params, M;
                metric = "A0",
                step_control = "MM",
                retraction = "newton"
            )
# Print final results
    println(result_summary)

# Plot density
    plot_density_2D(u_final, mesh, params)
