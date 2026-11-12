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
    β2::Float64
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
    u2::Vector{ComplexF64}
    u1::Vector{ComplexF64}
    u0::Vector{ComplexF64}
    um1::Vector{ComplexF64}
    um2::Vector{ComplexF64}
end

struct SpinMatrices{F, Dim, M <: SMatrix{Dim, Dim, ComplexF64}}
    Sx::M 
    Sy::M 
    Sz::M
    A00::M
end

function SpinMatrices(F::Int)
    Dim = 2 * F + 1

    if F == 2
        sqrt32 = sqrt(1.5)
        inv_sqrt5 = 1.0 / sqrt(5.0)

        Sx_s = @SMatrix ComplexF64[
            0.0   1.0   0.0   0.0   0.0;
            1.0   0.0   sqrt32   0.0   0.0;
            0.0   sqrt32   0.0   sqrt32   0.0;
            0.0   0.0   sqrt32   0.0   1.0;
            0.0   0.0   0.0   1.0   0.0
        ]

        Sy_s = @SMatrix ComplexF64[
            0.0     1.0im    0.0      0.0     0.0;
           -1.0im   0.0      sqrt32*1.0im 0.0     0.0;
            0.0    -sqrt32*1.0im 0.0      sqrt32*1.0im 0.0;
            0.0     0.0     -sqrt32*1.0im 0.0     1.0im;
            0.0     0.0      0.0     -1.0im   0.0
        ]

        Sz_s = @SMatrix ComplexF64[
             2.0  0.0  0.0   0.0   0.0;
             0.0  1.0  0.0   0.0   0.0;
             0.0  0.0  0.0   0.0   0.0;
             0.0  0.0  0.0  -1.0   0.0;
             0.0  0.0  0.0   0.0  -2.0
        ]

        A00_s = @SMatrix ComplexF64[
            0.0         0.0         0.0         0.0         1.0 * inv_sqrt5;
            0.0         0.0         0.0        -1.0 * inv_sqrt5 0.0;
            0.0         0.0         1.0 * inv_sqrt5 0.0         0.0;
            0.0        -1.0 * inv_sqrt5 0.0         0.0         0.0;
            1.0 * inv_sqrt5 0.0         0.0         0.0         0.0
        ]

        M = typeof(Sx_s)
        return SpinMatrices{F, Dim, M}(Sx_s, Sy_s, Sz_s, A00_s)
    else
        error("Unsupported spin value F = $F")
    end
end
function to_vec!(out::Vector{ComplexF64}, u::SpinorState)
    n = length(u.u2)
    @inbounds begin
        out[1:n]        .= u.u2
        out[n+1:2n]     .= u.u1
        out[2n+1:3n]    .= u.u0
        out[3n+1:4n]    .= u.um1
        out[4n+1:5n]    .= u.um2
    end
    return out
end

function from_vec!(u_out::SpinorState, v::AbstractVector)
    n = div(length(v),5)
    @inbounds begin
        u_out.u2  .= @view(v[1:n])
        u_out.u1  .= @view(v[n+1:2n])
        u_out.u0  .= @view(v[2n+1:3n])
        u_out.um1 .= @view(v[3n+1:4n])
        u_out.um2 .=@view(v[4n+1:5n])
    end
    return u_out
end

function Base.:+(u::SpinorState, v::SpinorState)
    sum_spinor_state = SpinorState(u.u2 + v.u2, u.u1 + v.u1, u.u0 + v.u0, u.um1 + v.um1, u.um2 + v.um2)
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

function spin2_energy(u, mesh, params, spin_mats)
    u2 = u.u2
    u1 = u.u1
    u0 = u.u0
    um1 = u.um1
    um2 = u.um2

    Sx = spin_mats.Sx
    Sy = spin_mats.Sy
    Sz = spin_mats.Sz
    A00 = spin_mats.A00

    V_fun = params.V_fun
    order_finite_element = params.order_finite_element
    x_mesh = mesh.x
    T = mesh.T
    bn = mesh.bn
    global_to_interior = mesh.global_to_interior

    no_of_basis = (order_finite_element == 1) ? 3 : 6

    β0 = params.β0
    β1 = params.β1
    β2 = params.β2
    q = params.q
    p = params.p

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

    E_kin = 0.0
    E_pot = 0.0
    E_zeeman = 0.0
    E_rho = 0.0
    E_spin = 0.0
    E_spin_singlet = 0.0

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

        u2_loc  = MVector{6, ComplexF64}(undef)
        u1_loc  = MVector{6, ComplexF64}(undef)
        u0_loc  = MVector{6, ComplexF64}(undef)
        um1_loc = MVector{6, ComplexF64}(undef)
        um2_loc  = MVector{6, ComplexF64}(undef)

        for i in 1:no_of_basis
            node_i = T[k, i]
            if bn[node_i] == 0
                gi = global_to_interior[node_i]
                u2_loc[i]  = ComplexF64(u2[gi])
                u1_loc[i]  = ComplexF64(u1[gi])
                u0_loc[i]  = ComplexF64(u0[gi])
                um1_loc[i] = ComplexF64(um1[gi])
                um2_loc[i]  = ComplexF64(um2[gi])
            else
                u2_loc[i]  = 0.0 + 0.0im
                u1_loc[i]  = 0.0 + 0.0im
                u0_loc[i]  = 0.0 + 0.0im
                um1_loc[i] = 0.0 + 0.0im
                um2_loc[i]  = 0.0 + 0.0im
            end
        end

        for qpt in 1:7
            xi = x_quad[qpt]
            wq = w[qpt]

            if order_finite_element == 1
                phi = SVector(
                    -xi[1] - xi[2] + 1.0,
                    xi[1],
                    xi[2]
                )
                grad_phi = SVector(
                    SVector(-1.0, -1.0),
                    SVector( 1.0,  0.0),
                    SVector( 0.0,  1.0)
                )
            else
                phi = SVector(
                    (1.0 - xi[1] - xi[2]) * (1.0 - 2.0 * xi[1] - 2.0 * xi[2]),
                    4.0 * xi[1] * (1.0 - xi[1] - xi[2]),
                    xi[1] * (2.0 * xi[1] - 1.0),
                    4.0 * xi[1] * xi[2],
                    xi[2] * (2.0 * xi[2] - 1.0),
                    4.0 * xi[2] * (1.0 - xi[1] - xi[2])
                )
                grad_phi = SVector(
                    SVector(4.0 * xi[1] + 4.0 * xi[2] - 3.0, 4.0 * xi[1] + 4.0 * xi[2] - 3.0),
                    SVector(-4.0 * (2.0 * xi[1] + xi[2] - 1.0), -4.0 * xi[1]),
                    SVector(4.0 * xi[1] - 1.0, 0.0),
                    SVector(4.0 * xi[2], 4.0 * xi[1]),
                    SVector(0.0, 4.0 * xi[2] - 1.0),
                    SVector(-4.0 * xi[2], -4.0 * (xi[1] + 2.0 * xi[2] - 1.0))
                )
            end

            x_phy = J11 * xi[1] + J12 * xi[2] + z1[1]
            y_phy = J21 * xi[1] + J22 * xi[2] + z1[2]
            V_val = V_fun(SVector(x_phy, y_phy))

            uq2  = 0.0 + 0.0im
            uq1  = 0.0 + 0.0im
            uq0  = 0.0 + 0.0im
            uqm1 = 0.0 + 0.0im
            uqm2  = 0.0 + 0.0im

            g_u2  = SVector(0.0 + 0.0im, 0.0 + 0.0im)
            g_u1  = SVector(0.0 + 0.0im, 0.0 + 0.0im)
            g_u0  = SVector(0.0 + 0.0im, 0.0 + 0.0im)
            g_um1 = SVector(0.0 + 0.0im, 0.0 + 0.0im)
            g_um2  = SVector(0.0 + 0.0im, 0.0 + 0.0im)

            for i in 1:no_of_basis
                phi_i = phi[i]
                uq2  += u2_loc[i]  * phi_i
                uq1  += u1_loc[i]  * phi_i
                uq0  += u0_loc[i]  * phi_i
                uqm1 += um1_loc[i] * phi_i
                uqm2 += um2_loc[i] * phi_i

                g_phys = SVector(
                    iJ11 * grad_phi[i][1] + iJ12 * grad_phi[i][2],
                    iJ21 * grad_phi[i][1] + iJ22 * grad_phi[i][2]
                )
                
                g_u2  += u2_loc[i]  * g_phys
                g_u1  += u1_loc[i]  * g_phys
                g_u0  += u0_loc[i]  * g_phys
                g_um1 += um1_loc[i] * g_phys
                g_um2  += um2_loc[i]  * g_phys
            end

            u_vec = SVector{5, ComplexF64}(uq2, uq1, uq0, uqm1, uqm2)

            grad_sq = real(dot(g_u2, g_u2))  + real(dot(g_u1, g_u1) + dot(g_u0, g_u0) + dot(g_um1, g_um1)) + real(dot(g_um2, g_um2))

            rho = real(dot(u_vec, u_vec))

            Fx = real(dot(u_vec, Sx * u_vec))
            Fy = real(dot(u_vec, Sy * u_vec))
            Fz = real(dot(u_vec, Sz * u_vec))
            F_sq = Fx^2 + Fy^2 + Fz^2

            F_singlet    = sum(u_vec[i] * A00[i, j] * u_vec[j] for i in 1:5, j in 1:5)
            F_singlet_sq = abs2(F_singlet)

            fac = wq * detBT

            E_kin    += fac * grad_sq
            E_pot    += fac * V_val * rho
            E_zeeman += fac * (
                (4.0 * q - 2.0 * p) * abs2(uq2) +
                (q - p) * abs2(uq1) + 
                (q + p) * abs2(uqm1) + 
                (4.0 * q + 2.0 * p) * abs2(uqm2)
                )
            E_rho    += fac * (rho^2)
            E_spin   += fac * F_sq
            E_spin_singlet += fac * F_singlet_sq
        end
    end

    E = 0.5 * E_kin + E_pot + E_zeeman + 0.5 * β0 * E_rho + 0.5 * β1 * E_spin + 0.5 * β2 * E_spin_singlet
    return E
end

struct Workspace
    g_out::SpinorState
    chunk_starts::Vector{Int}
    chunk_ends::Vector{Int}
    out_buffers::Vector{Vector{ComplexF64}}
    Mu2::Vector{ComplexF64}
    Mu1::Vector{ComplexF64}
    Mu0::Vector{ComplexF64}
    Mum1::Vector{ComplexF64}
    Mum2::Vector{ComplexF64}

    res_u2::Vector{ComplexF64} 
    res_u1::Vector{ComplexF64}   
    res_u0::Vector{ComplexF64}   
    res_um1::Vector{ComplexF64}
    res_um2::Vector{ComplexF64} 

    u2_evals::Vector{Vector{ComplexF64}}
    u1_evals::Vector{Vector{ComplexF64}}
    u0_evals::Vector{Vector{ComplexF64}}
    um1_evals::Vector{Vector{ComplexF64}}
    um2_evals::Vector{Vector{ComplexF64}}
    
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
    n_flat = 5 * n_comp
    no_of_basis = (order_finite_element == 1) ? 3 : 6

    g_out = SpinorState(zeros(ComplexF64, n_comp), zeros(ComplexF64, n_comp),zeros(ComplexF64, n_comp), zeros(ComplexF64, n_comp), zeros(ComplexF64, n_comp))
    Mu2  = zeros(ComplexF64, n_comp)
    Mu1  = zeros(ComplexF64, n_comp)
    Mu0  = zeros(ComplexF64, n_comp)
    Mum1 = zeros(ComplexF64, n_comp)
    Mum2  = zeros(ComplexF64, n_comp)
    
    res_u2  = zeros(ComplexF64, n_comp)
    res_u1  = zeros(ComplexF64, n_comp)
    res_u0  = zeros(ComplexF64, n_comp)
    res_um1 = zeros(ComplexF64, n_comp)
    res_um2  = zeros(ComplexF64, n_comp)
    
    u2_evals  = [Vector{ComplexF64}(undef, no_of_basis) for _ in 1:n_threads]
    u1_evals  = [Vector{ComplexF64}(undef, no_of_basis) for _ in 1:n_threads]
    u0_evals  = [Vector{ComplexF64}(undef, no_of_basis) for _ in 1:n_threads]
    um1_evals = [Vector{ComplexF64}(undef, no_of_basis) for _ in 1:n_threads]
    um2_evals  = [Vector{ComplexF64}(undef, no_of_basis) for _ in 1:n_threads]
    
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
    sobolev_grad   = SpinorState(zeros(ComplexF64, n_comp), zeros(ComplexF64, n_comp), zeros(ComplexF64, n_comp), zeros(ComplexF64, n_comp), zeros(ComplexF64, n_comp))
    ∇_A_N = SpinorState(zeros(ComplexF64, n_comp), zeros(ComplexF64, n_comp), zeros(ComplexF64, n_comp), zeros(ComplexF64, n_comp), zeros(ComplexF64, n_comp))
    ∇_A_M = SpinorState(zeros(ComplexF64, n_comp), zeros(ComplexF64, n_comp), zeros(ComplexF64, n_comp), zeros(ComplexF64, n_comp), zeros(ComplexF64, n_comp))
    u_proj =  SpinorState(zeros(ComplexF64, n_comp), zeros(ComplexF64, n_comp), zeros(ComplexF64, n_comp), zeros(ComplexF64, n_comp), zeros(ComplexF64, n_comp))

    chunk_size = ceil(Int, n_elements / n_threads)
    c_starts = [(i - 1) * chunk_size + 1 for i in 1:n_threads]
    c_ends   = [min(i * chunk_size, n_elements) for i in 1:n_threads]

    active = [i for i in 1:n_threads if c_starts[i] <= c_ends[i]]
    MLu = SpinorState(zeros(ComplexF64, n_comp), zeros(ComplexF64, n_comp), zeros(ComplexF64, n_comp), zeros(ComplexF64, n_comp), zeros(ComplexF64, n_comp))

    return Workspace(
        g_out, 
        c_starts[active], 
        c_ends[active], 
        out_buffers[active], 
        Mu2, Mu1, Mu0, Mum1, Mum2,
        res_u2, res_u1, res_u0, res_um1, res_um2,
        u2_evals[active], u1_evals[active], u0_evals[active], um1_evals[active], um2_evals[active],
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

    q = params.q
    p = params.p

    n_comp = div(length(x), 5)
    no_of_basis = (order == 1) ? 3 : 6

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

    @inbounds for qpt in 1:7
        xq1, xq2 = x_quad[qpt][1], x_quad[qpt][2]
        if order == 1
            phi[1, qpt] = 1.0 - xq1 - xq2
            phi[2, qpt] = xq1
            phi[3, qpt] = xq2

            grad_phi[1,  qpt] = -1.0; grad_phi[2, qpt] = -1.0
            grad_phi[1, 7 + qpt] =  1.0; grad_phi[2, 7 + qpt] =  0.0
            grad_phi[1, 2*7 + qpt] =  0.0; grad_phi[2, 2*7 + qpt] =  1.0
        else
            phi[1, qpt] = (1 - xq1 - xq2) * (1 - 2*xq1 - 2*xq2)
            phi[2, qpt] = 4 * xq1 * (1 - xq1 - xq2)
            phi[3, qpt] = xq1 * (2*xq1 - 1)
            phi[4, qpt] = 4 * xq1 * xq2
            phi[5, qpt] = xq2 * (2*xq2 - 1)
            phi[6, qpt] = 4 * xq2 * (1 - xq1 - xq2)

            grad_phi[1, qpt] = 4*xq1 + 4*xq2 - 3; grad_phi[2, qpt] = 4*xq1 + 4*xq2 - 3
            grad_phi[1, 7 + qpt] = -4*(2*xq1 + xq2 - 1); grad_phi[2, 7 + qpt] = -4*xq1
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

                    for q in 1:7
                        phi_i = phi[i, q]
                        phi_j = phi[j, q]

                        gi_x = iJ11 * grad_phi[1, (i-1)*7 + q] + iJ12 * grad_phi[2, (i-1)*7 + q]
                        gi_y = iJ21 * grad_phi[1, (i-1)*7 + q] + iJ22 * grad_phi[2, (i-1)*7 + q]

                        gj_x = iJ11 * grad_phi[1, (j-1)*7 + q] + iJ12 * grad_phi[2, (j-1)*7 + q]
                        gj_y = iJ21 * grad_phi[1, (j-1)*7 + q] + iJ22 * grad_phi[2, (j-1)*7 + q]

                        xq, yq = x_quad[q]
                        x_phy = J11 * xq + J12 * yq + z1[1]
                        y_phy = J21 * xq + J22 * yq + z1[2]

                        #FACTOR 2
                        m_q = 2 * phi_i * phi_j
                        stiff_q = (gi_x * gj_x + gi_y * gj_y)
                        pot_q = V_fun(SVector(x_phy, y_phy)) * m_q

                        base_ij += w[q] * (stiff_q + pot_q)
                        mass_ij += w[q] * m_q
                    end

                    base_ij *= detBT
                    mass_ij *= detBT

                    a0_u2 = base_ij + (4*q - 2*p) * mass_ij
                    a0_u1  = base_ij + (q - p)  * mass_ij
                    a0_u0  = base_ij 
                    a0_um1 = base_ij + (q + p) * mass_ij
                    a0_um2 = base_ij + (4*q + 2*p) * mass_ij

                    y_loc[eq_i] += a0_u2  * x[eq_j]
                    y_loc[n_comp  + eq_i] += a0_u1  * x[n_comp  + eq_j]
                    y_loc[2 * n_comp + eq_i] += a0_u0 * x[2 * n_comp + eq_j]
                    y_loc[3 * n_comp + eq_i] += a0_um1 * x[3 * n_comp + eq_j]
                    y_loc[4 * n_comp + eq_i] += a0_um2 * x[4 * n_comp + eq_j]
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

function apply_A_with_spin_parallel!(
    out::Vector{ComplexF64}, vec::Vector{ComplexF64},
    u2, u1, u0, um1, um2,  mesh, params,  n_comp::Int, no_of_basis::Int, 
    w::AbstractVector{Float64}, phi_in_quad_point::AbstractMatrix{Float64}, 
    β0::Float64, β1::Float64, β2::Float64, ws
)
    chunk_starts = ws.chunk_starts 
    chunk_ends   = ws.chunk_ends 
    out_buffers  = ws.out_buffers 
    num_chunks = length(chunk_starts)
    u2_evals   = [Vector{ComplexF64}(undef, no_of_basis) for _ in 1:num_chunks]
    u1_evals   = [Vector{ComplexF64}(undef, no_of_basis) for _ in 1:num_chunks]
    u0_evals   = [Vector{ComplexF64}(undef, no_of_basis) for _ in 1:num_chunks]
    um1_evals  = [Vector{ComplexF64}(undef, no_of_basis) for _ in 1:num_chunks]
    um2_evals   = [Vector{ComplexF64}(undef, no_of_basis) for _ in 1:num_chunks]

    apply_A0_parallel!(out, vec, mesh, params, ws)

    # Reset thread-local accumulators
    @threads for cid in 1:length(chunk_starts)
        fill!(out_buffers[cid], 0.0)
    end

    x_nodes = mesh.x
    T = mesh.T
    bn = mesh.bn
    g2i = mesh.global_to_interior

    sqrt32 = sqrt(1.5)
    inv_sqrt5 = 1.0 / sqrt(5.0)

    # Parallel Element Loop
    @threads for cid in 1:length(chunk_starts)
        c_start  = chunk_starts[cid]
        c_end    = chunk_ends[cid]
        out_loc  = out_buffers[cid]
        u2_eval  = u2_evals[cid]
        u1_eval  = u1_evals[cid]
        u0_eval  = u0_evals[cid]
        um1_eval = um1_evals[cid]
        um2_eval = um2_evals[cid]

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
                    u2_eval[i]  = ComplexF64(u2[idx])
                    u1_eval[i]  = ComplexF64(u1[idx])
                    u0_eval[i]  = ComplexF64(u0[idx])
                    um1_eval[i] = ComplexF64(um1[idx])
                    um2_eval[i] = ComplexF64(um2[idx])
                else
                    u2_eval[i]  = 0.0im
                    u1_eval[i]  = 0.0im
                    u0_eval[i]  = 0.0im
                    um1_eval[i] = 0.0im
                    um2_eval[i] = 0.0im
                end
            end

            rho_q    = MVector{7, Float64}(undef)
            A_q     = MVector{7, ComplexF64}(undef)
            Fz_q     = MVector{7, Float64}(undef)
            A00_val_q = MVector{7, ComplexF64}(undef)

            for l in 1:7
                s_u2, s_u1, s_u0, s_um1, s_um2 = 0.0im, 0.0im, 0.0im, 0.0im, 0.0im
                for b in 1:no_of_basis
                    p_val = phi_in_quad_point[b, l]
                    s_u2  += p_val * u2_eval[b]
                    s_u1  += p_val * u1_eval[b]
                    s_u0  += p_val * u0_eval[b]
                    s_um1 += p_val * um1_eval[b]
                    s_um2 += p_val * um2_eval[b]
                end
                
                rho_q[l] = abs2(s_u2) + abs2(s_u1) + abs2(s_u0) + abs2(s_um1) + abs2(s_um2)
                A_q[l] = 2.0 * (s_u1 * conj(s_u2) + sqrt32 * (s_u0 * conj(s_u1) + s_um1 * conj(s_u0)) + s_um2 * conj(s_um1))
                Fz_q[l] = 2.0 * abs2(s_u2) + abs2(s_u1) - abs2(s_um1) - 2.0 * abs2(s_um2)
                
                A00_val_q[l] = inv_sqrt5 * (2.0 * s_um2 * s_u2 - 2.0 * s_um1 * s_u1 + s_u0^2)
            end

            for j in 1:no_of_basis
                node_j = T[k, j]
                idx_j = g2i[node_j]
                
                if idx_j > 0 && bn[node_j] == 0

                    v_j2  = ComplexF64(vec[idx_j])
                    v_j1  = ComplexF64(vec[idx_j + n_comp])
                    v_j0  = ComplexF64(vec[idx_j + 2*n_comp])
                    v_jm1 = ComplexF64(vec[idx_j + 3*n_comp])
                    v_jm2 = ComplexF64(vec[idx_j + 4*n_comp])


                    for i in 1:no_of_basis
                        node_i = T[k, i]
                        idx_i = g2i[node_i]
                        
                        if idx_i > 0 && bn[node_i] == 0
                            out_spin_2, out_spin_1, out_spin_0, out_spin_m1, out_spin_m2 = 0.0im, 0.0im, 0.0im, 0.0im, 0.0im

                            for l in 1:7
                                fac = phi_in_quad_point[i, l] * phi_in_quad_point[j, l]
                                fz = Fz_q[l]
                                A = A_q[l]
                                A_bar = conj(A)
                                A00_val = A00_val_q[l]
                                
                                coeff_β0 = 2.0 * β0 * rho_q[l]
                                b0_2 = coeff_β0 * v_j2
                                b0_1 = coeff_β0 * v_j1
                                b0_0 = coeff_β0 * v_j0
                                b0_m1 = coeff_β0 * v_jm1
                                b0_m2 = coeff_β0 * v_jm2

                                b1_2  = 2.0 * β1 * (A_bar * v_j1 + 2.0 * fz * v_j2)
                                b1_1  = 2.0 * β1 * (A * v_j2 + sqrt32 * A_bar * v_j0 + fz * v_j1)
                                b1_0  = 2.0 * β1 * (sqrt32 * A_bar * v_jm1 + sqrt32 * A * v_j1)
                                b1_m1 = 2.0 * β1 * (sqrt32 * A * v_j0 + A_bar * v_jm2 - fz * v_jm1)
                                b1_m2 = 2.0 * β1 * (A * v_jm1 - 2.0 * fz * v_jm2)                             

                                coeff_β2 = 2.0 * β2 * inv_sqrt5 * A00_val

                                b2_2  = coeff_β2 * conj(v_jm2)
                                b2_1  = -coeff_β2 * conj(v_jm1)
                                b2_0  = coeff_β2 * conj(v_j0)
                                b2_m1 = -coeff_β2* conj(v_j1)
                                b2_m2 = coeff_β2 * conj(v_j2)

                                weight = w[l] * fac * detBT

                                out_spin_2 += weight * (b0_2  + b1_2 + b2_2)
                                out_spin_1  += weight * (b0_1  + b1_1 + b2_1)
                                out_spin_0  += weight * (b0_0  + b1_0 + b2_0)
                                out_spin_m1 += weight * (b0_m1 + b1_m1 + b2_m1)
                                out_spin_m2 += weight * (b0_m2  + b1_m2 + b2_m2)
                            end

                            out_loc[idx_i]                          += out_spin_2
                            out_loc[idx_i + n_comp]                 += out_spin_1
                            out_loc[idx_i + 2*n_comp]               += out_spin_0
                            out_loc[idx_i + 3*n_comp]               += out_spin_m1
                            out_loc[idx_i + 4*n_comp]               += out_spin_m2
                        end
                    end
                end
            end
        end
    end

    num_chunks = length(chunk_starts)
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
    u2, u1, u0, um1, um2, mesh, params, n_comp::Int, no_of_basis::Int, 
    w::AbstractVector{Float64}, phi_in_quad_point::AbstractMatrix{Float64}, 
    β0::Float64, β1::Float64, β2::Float64, ws
)   
    chunk_starts = ws.chunk_starts 
    chunk_ends   = ws.chunk_ends 
    out_buffers  = ws.out_buffers 
    num_chunks = length(chunk_starts)

    u2_evals   = [Vector{ComplexF64}(undef, no_of_basis) for _ in 1:num_chunks]
    u1_evals   = [Vector{ComplexF64}(undef, no_of_basis) for _ in 1:num_chunks]
    u0_evals   = [Vector{ComplexF64}(undef, no_of_basis) for _ in 1:num_chunks]
    um1_evals  = [Vector{ComplexF64}(undef, no_of_basis) for _ in 1:num_chunks]
    um2_evals   = [Vector{ComplexF64}(undef, no_of_basis) for _ in 1:num_chunks]

    apply_A0_parallel!(out, vec, mesh, params, ws)

    # Reset thread local accumulators
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

        u2_eval  = u2_evals[cid]
        u1_eval  = u1_evals[cid]
        u0_eval  = u0_evals[cid]
        um1_eval = um1_evals[cid]
        um2_eval = um2_evals[cid]

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
                    u2_eval[i]  = ComplexF64(u2[idx])
                    u1_eval[i]  = ComplexF64(u1[idx])
                    u0_eval[i]  = ComplexF64(u0[idx])
                    um1_eval[i] = ComplexF64(um1[idx])
                    um2_eval[i]  = ComplexF64(um2[idx])
                else
                    u2_eval[i], u1_eval[i], u0_eval[i], um1_eval[i], um2_eval[i] = 0.0im, 0.0im, 0.0im, 0.0im, 0.0im
                end
            end

            rho_q = MVector{7, Float64}(undef)

            for l in 1:7
                s_u2, s_u1, s_u0, s_um1, s_um2 = 0.0im, 0.0im, 0.0im, 0.0im, 0.0im
                for b in 1:no_of_basis
                    p_val = phi_in_quad_point[b, l]

                    s_u2  += p_val * u2_eval[b]
                    s_u1  += p_val * u1_eval[b]
                    s_u0  += p_val * u0_eval[b]
                    s_um1 += p_val * um1_eval[b]
                    s_um2 += p_val * um2_eval[b]
                end
                
                rho_q[l] = abs2(s_u2) + abs2(s_u1) + abs2(s_u0) + abs2(s_um1) + abs2(s_um2)
            end

            for j in 1:no_of_basis
                node_j = T[k, j]
                idx_j = g2i[node_j]
                
                if idx_j > 0 && bn[node_j] == 0
                    v_j2  = ComplexF64(vec[idx_j])
                    v_j1  = ComplexF64(vec[idx_j + n_comp])
                    v_j0 = ComplexF64(vec[idx_j + 2*n_comp])
                    v_jm1 = ComplexF64(vec[idx_j + 3*n_comp])
                    v_jm2 = ComplexF64(vec[idx_j + 4*n_comp])

                    for i in 1:no_of_basis
                        node_i = T[k, i]
                        idx_i = g2i[node_i]
                        
                        if idx_i > 0 && bn[node_i] == 0
                            out2  = 0.0im
                            out1  = 0.0im
                            out0  = 0.0im
                            outm1 = 0.0im
                            outm2  = 0.0im

                            for l in 1:7
                                fac = phi_in_quad_point[i, l] * phi_in_quad_point[j, l]
                                rho = rho_q[l]

                                coeff_β0 = 2.0 * β0 * rho
                                b0_2  = coeff_β0 * v_j2
                                b0_1  = coeff_β0 * v_j1
                                b0_0  = coeff_β0 * v_j0
                                b0_m1 = coeff_β0 * v_jm1
                                b0_m2  = coeff_β0 * v_jm2

                                weight = w[l] * fac * detBT
                                out2  += weight * b0_2
                                out1  += weight * b0_1 
                                out0  += weight * b0_0  
                                outm1 += weight * b0_m1 
                                outm2  += weight * b0_m2 
                            end

                            out_loc[idx_i]                          += out2
                            out_loc[idx_i + n_comp]                 += out1
                            out_loc[idx_i + 2*n_comp]               += out0
                            out_loc[idx_i + 3*n_comp]                 += outm1
                            out_loc[idx_i + 4*n_comp]               += outm2
                        end
                    end
                end
            end
        end
    end

    num_chunks = length(chunk_starts)
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
    u2, u1, u0, um1, um2, mesh, params, n_comp::Int, no_of_basis::Int, 
     w::AbstractVector{Float64}, phi_in_quad_point::AbstractMatrix{Float64}, 
    β0::Float64, β1::Float64, β2::Float64, ws
)

    chunk_starts = ws.chunk_starts 
    chunk_ends   = ws.chunk_ends 
    out_buffers  = ws.out_buffers 
    num_chunks = length(chunk_starts)
    u2_evals   = [Vector{ComplexF64}(undef, no_of_basis) for _ in 1:num_chunks]
    u1_evals   = [Vector{ComplexF64}(undef, no_of_basis) for _ in 1:num_chunks]
    u0_evals   = [Vector{ComplexF64}(undef, no_of_basis) for _ in 1:num_chunks]
    um1_evals  = [Vector{ComplexF64}(undef, no_of_basis) for _ in 1:num_chunks]
    um2_evals   = [Vector{ComplexF64}(undef, no_of_basis) for _ in 1:num_chunks]

    apply_A0_parallel!(out, vec, mesh, params, ws)

    # Reset thread local accumulators
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
        u2_eval  = u2_evals[cid]
        u1_eval  = u1_evals[cid]
        u0_eval  = u0_evals[cid]
        um1_eval = um1_evals[cid]
        um2_eval = um2_evals[cid]

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
                    u2_eval[i]  = ComplexF64(u2[idx])
                    u1_eval[i]  = ComplexF64(u1[idx])
                    u0_eval[i]  = ComplexF64(u0[idx])
                    um1_eval[i] = ComplexF64(um1[idx])
                    um2_eval[i] = ComplexF64(um2[idx])
                else
                    u2_eval[i]  = 0.0im
                    u1_eval[i]  = 0.0im
                    u0_eval[i]  = 0.0im
                    um1_eval[i] = 0.0im
                    um2_eval[i] = 0.0im
                end
            end

            rho_q      = MVector{7, Float64}(undef)
            abs2_u2_q  = MVector{7, Float64}(undef)
            abs2_u1_q  = MVector{7, Float64}(undef)
            abs2_u0_q  = MVector{7, Float64}(undef)
            abs2_um1_q = MVector{7, Float64}(undef)
            abs2_um2_q  = MVector{7, Float64}(undef)

            for l in 1:7
                s_u2, s_u1, s_u0, s_um1, s_um2 = 0.0im, 0.0im, 0.0im, 0.0im, 0.0im
                for b in 1:no_of_basis
                    p_val = phi_in_quad_point[b, l]
                    s_u2  += p_val * u2_eval[b]
                    s_u1  += p_val * u1_eval[b]
                    s_u0  += p_val * u0_eval[b]
                    s_um1 += p_val * um1_eval[b]
                    s_um2 += p_val * um2_eval[b]
                end
                abs2_u2_q[l]  = abs2(s_u2)
                abs2_u1_q[l]  = abs2(s_u1)
                abs2_u0_q[l]  = abs2(s_u0)
                abs2_um1_q[l] = abs2(s_um1)
                abs2_um2_q[l]  = abs2(s_um2)
                rho_q[l]      =  abs2_u2_q[l] + abs2_u1_q[l] + abs2_u0_q[l] + abs2_um1_q[l] + abs2_um2_q[l]
            end

            for j in 1:no_of_basis
                node_j = T[k, j]
                idx_j = g2i[node_j]
                
                if idx_j > 0 && bn[node_j] == 0
                    v_j2  = ComplexF64(vec[idx_j])
                    v_j1  = ComplexF64(vec[idx_j + n_comp])
                    v_j0  = ComplexF64(vec[idx_j + 2*n_comp])
                    v_jm1 = ComplexF64(vec[idx_j + 3*n_comp])
                    v_jm2 = ComplexF64(vec[idx_j + 4*n_comp])

                    for i in 1:no_of_basis
                        node_i = T[k, i]
                        idx_i = g2i[node_i]
                        
                        if idx_i > 0 && bn[node_i] == 0
                            out2 = 0.0im
                            out1  = 0.0im
                            out0  = 0.0im
                            outm1 = 0.0im
                            outm2 = 0.0im

                            for l in 1:7
                                fac = phi_in_quad_point[i, l] * phi_in_quad_point[j, l]
                                rho = rho_q[l]                       
                                
                                abs2_u2  = abs2_u2_q[l]
                                abs2_u1  = abs2_u1_q[l]
                                abs2_u0  = abs2_u0_q[l]
                                abs2_um1 = abs2_um1_q[l]
                                abs2_um2 = abs2_um2_q[l]
                                

                                coeff_β0 = 2.0 * β0 * rho

                                Fz = 2.0 * abs2_u2 + abs2_u1 - abs2_um1 - 2.0 * abs2_um2

                                coeff_b1_2  = 2.0 * β1 * (2.0 * abs2_u1 + 2.0 * Fz) 
                                coeff_b1_1  = 2.0 * β1 * (2.0 * abs2_u2 + 3.0 * abs2_u0 + Fz)
                                coeff_b1_0  = 2.0 * β1 * (3.0 * abs2_u1 + 3.0 * abs2_um1)
                                coeff_b1_m1 = 2.0 * β1 * (2.0 * abs2_um2 + 3.0 * abs2_u0 - Fz)
                                coeff_b1_m2 = 2.0 * β1 * (2.0 * abs2_um1 - 2.0 * Fz) 

                                coeff_b2_2  = 0.8 * β2 * abs2_um2
                                coeff_b2_1  = 0.8 * β2 * abs2_um1
                                coeff_b2_0  = 0.4 * β2 * abs2_u0
                                coeff_b2_m1 = 0.8 * β2 * abs2_u1
                                coeff_b2_m2 = 0.8 * β2 * abs2_u2

                                weight = w[l] * fac * detBT
                                out2  += weight * (coeff_β0 + coeff_b1_2  + coeff_b2_2)  * v_j2
                                out1  += weight * (coeff_β0 + coeff_b1_1  + coeff_b2_1)  * v_j1
                                out0  += weight * (coeff_β0 + coeff_b1_0  + coeff_b2_0)  * v_j0
                                outm1 += weight * (coeff_β0 + coeff_b1_m1 + coeff_b2_m1) * v_jm1
                                outm2 += weight * (coeff_β0 + coeff_b1_m2 + coeff_b2_m2) * v_jm2
                            end

                            out_loc[idx_i]                          += out2
                            out_loc[idx_i + n_comp]                 += out1
                            out_loc[idx_i + 2*n_comp]               += out0
                            out_loc[idx_i + 3*n_comp]               += outm1
                            out_loc[idx_i + 4*n_comp]               += outm2
                        end
                    end
                end
            end
        end
    end

    num_chunks = length(chunk_starts)
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
    u2, u1, u0, um1, um2, mesh, params,
    n_comp::Int, no_of_basis::Int,
    w::AbstractVector{Float64},
    phi_in_quad_point::AbstractMatrix{Float64},
    β0::Float64, β1::Float64, β2::Float64, ws, metric
)
    if metric == "spin"
        apply_A_with_spin_parallel!(
            out, vec, u2, u1, u0, um1, um2, mesh, params,
            n_comp, no_of_basis, w, phi_in_quad_point,
            β0, β1, β2, ws
        )
    elseif metric == "spin-diagonal"
        apply_A_spin_diagonal_parallel!(
            out, vec, u2, u1, u0, um1, um2, mesh, params,
            n_comp, no_of_basis, w, phi_in_quad_point,
            β0, β1, β2, ws
        )
    elseif metric == "density"
        apply_A_density_parallel!(
            out, vec, u2, u1, u0, um1, um2, mesh, params,
            n_comp, no_of_basis, w, phi_in_quad_point,
            β0, β1, β2, ws
        )
    elseif metric == "A0"
        apply_A0_parallel!(out, vec, mesh, params, ws)
    else
        error("Unknown metric: $metric")
    end

    return nothing
end


function cg_A_parallel!( #output x = A\b
    x_state::SpinorState,   # Pre-allocated output SpinorState
    mesh, params, u, b_state, max_iter, tol, metric, P, ws::Workspace
)
    β0 = params.β0 
    β1 = params.β1
    β2 = params.β2
    
    n_comp = length(u.um1)
    n = 5 * n_comp
    
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

    # Preconditioning step 0: z_0 = P \ r_0
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

    order = params.order_finite_element
    no_of_basis = (order == 1) ? 3 : 6

    phi_in_quad_point = MMatrix{6, 7, Float64}(undef)
    for qpt in 1:7
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

    um2, um1, u0, u1, u2 = u.um2, u.um1, u.u0, u.u1, u.u2

    k = 0
    @inbounds while r_norm >= tol && k < max_iter
        apply_A_parallel!(
                Ap_out, p, u2, u1, u0, um1, um2, mesh, params,
                n_comp, no_of_basis, w, phi_in_quad_point, β0, β1, β2,
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
        @show(tol)
        error("CG did not converge within $max_iter iterations (res_norm = $r_norm)")
    end

    from_vec!(x_state, x)
    return x_state
end

function frechet_derivative!(
    u::SpinorState, mesh, params, ws::Workspace
)

    out_buffers  = ws.out_buffers 
    u_vec = ws.u_vec 
    out = ws.out
    g = ws.g_out

    sqrt32 = sqrt(1.5)
    sqrt6 = sqrt(6)

    u2 = u.u2
    u1 = u.u1
    u0 = u.u0
    um1 = u.um1
    um2 = u.um2

    x   = mesh.x 
    T   = mesh.T 
    bn  = mesh.bn
    global_to_interior = mesh.global_to_interior

    order_finite_element = params.order_finite_element
    β0 = params.β0
    β1 = params.β1
    β2 = params.β2

    to_vec!(u_vec, u)

    for buf in out_buffers
        fill!(buf, 0.0)
    end
    fill!(out, 0.0)

    apply_A0_parallel!(out, u_vec, mesh, params, ws)
    from_vec!(g, out)

    g2 = g.u2
    g1  = g.u1
    g0  = g.u0
    gm1 = g.um1
    gm2 = g.um2

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
    no_of_basis = (order_finite_element == 1) ? 3 : 6

    @inbounds for k in axes(T, 1)
        has_interior = false
        for a_idx in 1:3
            if bn[T[k, a_idx]] == 0
                has_interior = true
                break
            end
        end
        !has_interior && continue

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

        um2_loc = MVector{6,ComplexF64}(undef)
        um1_loc = MVector{6,ComplexF64}(undef)
        u0_loc  = MVector{6,ComplexF64}(undef)
        u1_loc  = MVector{6,ComplexF64}(undef)
        u2_loc  = MVector{6,ComplexF64}(undef)
        fill!(um2_loc, 0.0im)
        fill!(um1_loc, 0.0im)
        fill!(u0_loc, 0.0im)
        fill!(u1_loc, 0.0im)
        fill!(u2_loc, 0.0im)

        for i in 1:no_of_basis
            node = T[k, i]
            if bn[node] == 0
                gi = global_to_interior[node]
                um2_loc[i] = um2[gi]
                um1_loc[i] = um1[gi]
                u0_loc[i]  = u0[gi]
                u1_loc[i]  = u1[gi]
                u2_loc[i] = u2[gi]
            end
        end
        nlm2_loc = MVector{6,ComplexF64}(undef)
        nlm1_loc = MVector{6,ComplexF64}(undef)
        nl0_loc  = MVector{6,ComplexF64}(undef)
        nl1_loc  = MVector{6,ComplexF64}(undef)
        nl2_loc = MVector{6,ComplexF64}(undef)
        fill!(nlm2_loc, 0.0im)
        fill!(nlm1_loc, 0.0im)
        fill!(nl0_loc, 0.0im)
        fill!(nl1_loc, 0.0im)
        fill!(nl2_loc, 0.0im)

        for qpt in 1:7
            xi = x_quad[qpt]
            wq = w[qpt]

            if order_finite_element == 1
                phi = SVector(-xi[1] - xi[2] + 1.0, xi[1], xi[2])
            else
                phi = SVector(
                    (1.0 - xi[1] - xi[2]) * (1.0 - 2.0 * xi[1] - 2.0 * xi[2]),
                    4.0 * xi[1] * (1.0 - xi[1] - xi[2]),
                    xi[1] * (2.0 * xi[1] - 1.0),
                    4.0 * xi[1] * xi[2],
                    xi[2] * (2.0 * xi[2] - 1.0),
                    4.0 * xi[2] * (1.0 - xi[1] - xi[2])
                )
            end

            uqm2 = 0.0im
            uqm1 = 0.0im
            uq0  = 0.0im
            uq1  = 0.0im
            uq2 = 0.0im

           for i in 1:no_of_basis
                uqm2 += phi[i] * um2_loc[i]
                uqm1 += phi[i] * um1_loc[i]
                uq0  += phi[i] * u0_loc[i]
                uq1  += phi[i] * u1_loc[i]
                uq2 += phi[i] * u2_loc[i]
           end

            abs_um2_sq = abs2(uqm2)
            abs_um1_sq = abs2(uqm1)
            abs_u0_sq  = abs2(uq0)
            abs_u1_sq  = abs2(uq1)
            abs_u2_sq = abs2(uq2)

            rho = abs_um2_sq + abs_um1_sq + abs_u0_sq + abs_u1_sq + abs_u2_sq
            
            f_rho_m2 = 2 * β0 * rho * uqm2
            f_rho_m1 = 2 * β0 * rho * uqm1
            f_rho_0  = 2 * β0 * rho * uq0
            f_rho_1  = 2 * β0 * rho * uq1
            f_rho_2 = 2 * β0 * rho * uq2

            Fz = 2.0 * abs_u2_sq + abs_u1_sq - abs_um1_sq - 2.0 * abs_um2_sq
            B = 2.0 * (uq1 * conj(uq2) + sqrt32 * uq0 * conj(uq1) + sqrt32 * uqm1 * conj(uq0) + uqm2 * conj(uqm1))
            B_conj = conj(B)
            A00_val = 2.0 * uqm2 * uq2 - 2.0 * uqm1 * uq1 + uq0^2
            
            f_spin_2 = 2.0 * β1 * (2 * Fz * uq2 + B_conj * uq1)
            f_spin_1 = 2.0 * β1 * (B * uq2 + Fz * uq1 + sqrt32 * B_conj * uq0)
            f_spin_0  = 2.0 * β1 * (sqrt32 * B * uq1 + sqrt32 * B_conj * uqm1)
            f_spin_m1  = 2.0 * β1 * (sqrt32 * B * uq0 - Fz * uqm1 + B_conj * uqm2)
            f_spin_m2 = 2.0 * β1 * (B * uqm1 - 2 * Fz * uqm2)

            f_spin_2_singlet = 2.0 * β2 * 0.2 * A00_val * conj(uqm2)
            f_spin_1_singlet = - 2.0 * β2 * 0.2 * A00_val * conj(uqm1)
            f_spin_0_singlet  = 2.0 * β2 * 0.2 * A00_val * conj(uq0)
            f_spin_m1_singlet  = -2.0 * β2 * 0.2 * A00_val * conj(uq1)
            f_spin_m2_singlet = 2.0 * β2 * 0.2 * A00_val * conj(uq2)

            f2  = f_rho_2  + f_spin_2 + f_spin_2_singlet
            f1  = f_rho_1  + f_spin_1 + f_spin_1_singlet
            f0  = f_rho_0  + f_spin_0 + f_spin_0_singlet
            fm1 = f_rho_m1 + f_spin_m1 + f_spin_m1_singlet
            fm2 = f_rho_m2 + f_spin_m2 + f_spin_m2_singlet  

            weight = wq * detBT
            for i in 1:no_of_basis
                nl2_loc[i] += weight * phi[i] * f2
                nl1_loc[i] += weight * phi[i] * f1
                nl0_loc[i]  += weight * phi[i] * f0
                nlm1_loc[i]  += weight * phi[i] * fm1
                nlm2_loc[i] += weight * phi[i] * fm2
            end
        end

        for i in 1:no_of_basis
            node = T[k, i]
            if bn[node] == 0
                gi = global_to_interior[node]
                gm2[gi] += nlm2_loc[i]
                gm1[gi] += nlm1_loc[i]
                g0[gi]  += nl0_loc[i]
                g1[gi]  += nl1_loc[i]
                g2[gi] += nl2_loc[i]
            end
        end
    end

    return g
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
  
    @. u_out.u2  = u.u2  + τ * dn.u2
    @. u_out.u1  = u.u1  + τ * dn.u1
    @. u_out.u0  = u.u0  + τ * dn.u0
    @. u_out.um1 = u.um1 + τ * dn.um1
    @. u_out.um2  = u.um2  + τ * dn.um2

    N_u2_out  = dot(u_out.u2,  M_final, u_out.u2)
    N_u1_out   = dot(u_out.u1,  M_final, u_out.u1)
    N_u0_out   = dot(u_out.u0,  M_final, u_out.u0)
    N_um1_out  = dot(u_out.um1, M_final, u_out.um1)
    N_um2_out   = dot(u_out.um2,  M_final, u_out.um2)


    if retraction == "newton"

        LuLu          = real(4 * dot(u.um2, M_final, u.um2) + dot(u.um1, M_final, u.um1) + dot(u.u1, M_final, u.u1) + 4 * dot(u.u2, M_final, u.u2) ) # (Lu,Lu)
        dot_u_out_u  = real(dot(u_out.um2, M_final, u.um2)  + dot(u_out.um1, M_final, u.um1) + dot(u_out.u0, M_final, u.u0) + dot(u_out.u1, M_final, u.u1) + dot(u_out.u2, M_final, u.u2) ) # (u_out, u)
        dot_u_out_Lu = real( 2 * dot(u_out.u2,  M_final, u.u2) + dot(u_out.u1,  M_final, u.u1)  - dot(u_out.um1, M_final, u.um1) - 2 * dot(u_out.um2, M_final, u.um2)) # (u_out, Lu)
        dot_Lu_out_Lu         = real(4 * dot(u_out.u2,  M_final, u.u2) + dot(u_out.u1,  M_final, u.u1)  + dot(u_out.um1, M_final, u.um1) + 4 * dot(u_out.um2,  M_final, u.um2)) # (Lu_out, Lu)

        u_out_sq = real(N_um2_out + N_um1_out + N_u0_out + N_u1_out + N_u2_out) #(u_out, u_out)
        dot_Lu_out_u_out= real(2 * N_u2_out + N_u1_out - N_um1_out - 2 * N_um2_out) # (Lu_out, u_out)
        dot_L_sq_u_Lu = real(dot(4 * u.u2, M_final, 2 * u.u2) + dot(u.u1, M_final, u.u1) + dot(u.um1, M_final, -u.um1) + dot(4 * u.um2, M_final, -2 * u.um2)
        )

         if LuLu-C_mag^2 <= 0 
            @show(LuLu-C_mag^2)
            error("Regularity check failed")
        end


        α = 0.0
        β = 0.0

        rhs_N = 10.0
        rhs_M = 10.0
        count = 1

        while (abs(rhs_N) > 1e-12 || abs(rhs_M) > 1e-12) && count <= 5000
            rhs_N =  α^2 + β^2 * LuLu - 2.0 * α * dot_u_out_u - 2.0 * β * dot_u_out_Lu + 2.0 * α * β * C_mag + u_out_sq - 1.0
            rhs_M =  α^2 * C_mag + β^2 * dot_L_sq_u_Lu - 2.0 * α * dot_u_out_Lu - 2.0 * β * dot_Lu_out_Lu + 2.0 * α * β * LuLu + dot_Lu_out_u_out  - C_mag

            partial_N_α = 2.0 * α - 2.0 * dot_u_out_u + 2.0 * β * C_mag
            partial_N_β = 2.0 * β * LuLu - 2.0 * dot_u_out_Lu + 2.0 * α * C_mag

            partial_M_α = 2.0 * α * C_mag - 2.0 * dot_u_out_Lu + 2.0 * β * LuLu
            partial_M_β = 2.0 * β * dot_L_sq_u_Lu- 2.0 * dot_Lu_out_Lu + 2.0 * α * LuLu

            detJ = (partial_N_α * partial_M_β - partial_N_β * partial_M_α)
            det_inv = 1.0 / detJ

            α -= det_inv * (partial_M_β * rhs_N - partial_N_β * rhs_M)
            β -= det_inv * (partial_N_α * rhs_M - partial_M_α * rhs_N)
            count += 1            
    end

        @. u_out.u2  -= (α + 2.0 * β) * u.u2
        @. u_out.u1  -= (α + 1.0 * β) * u.u1
        @. u_out.u0  -= α * u.u0
        @. u_out.um1 -= (α - 1.0 * β) * u.um1
        @. u_out.um2 -= (α - 2.0 * β) * u.um2

    elseif retraction == "inexact"

        m0 = N_u2_out + N_u1_out + N_u0_out + N_um1_out + N_um2_out
        m1 = 2.0 * N_u2_out + N_u1_out - N_um1_out - 2.0 * N_um2_out 
        m2 = 4.0 * N_u2_out+ N_u1_out + N_um1_out + 4.0 * N_um2_out

        a = (m0 * (C_mag * m0 - m1)) / (2.0 * (m0 * m2 - m1^2))
        denom = real(m0 + 2.0 * m1 * a + m2 * a^2)

        if denom < 0.0 
            error("Inexact projection not possible: negative denominator")
        end
        σ0  = 1.0 / sqrt(denom)
        σ2  = σ0 * (1.0 + 2.0 * a)
        σ1  = σ0 * (1.0 + a)
        σm1 = σ0 * (1.0 - a)
        σm2 = σ0 * (1.0 - 2.0 * a)

        @. u_out.u2  *= σ2
        @. u_out.u1  *= σ1
        @. u_out.u0  *= σ0
        @. u_out.um1 *= σm1
        @. u_out.um2 *= σm2

    elseif retraction == "closed-form"

        u_out_norm_sq = N_um2_out + N_um1_out + N_u0_out + N_u1_out + N_u2_out
        Lu_out_norm_sq = 4 * N_um2_out+ N_u1_out + N_um1_out + 4 * N_u2_out
        u_out_Lu_out = 2 * N_u2_out + N_u1_out - N_um1_out - 2 * N_um2_out

        denom = real(u_out_norm_sq * Lu_out_norm_sq - u_out_Lu_out^2)
        if denom <0 
            @show(denom)
            error("Negative sqrt")
        end

        σ_nom = zeros(5)

        for j in 1:5
            l = 3 - j  # 5 = (F+1)/2

            num = real(Lu_out_norm_sq -
                (C_mag + l) * u_out_Lu_out +
                C_mag * l * u_out_norm_sq)
            if num < 0
                @show(num)
                error("Negative sqrt")
            end

            σ_nom[j] = sqrt(num / denom)
        end

        @. u_out.u2  *= σ_nom[1]
        @. u_out.u1  *= σ_nom[2]
        @. u_out.u0  *= σ_nom[3]
        @. u_out.um1 *= σ_nom[4]
        @. u_out.um2  *= σ_nom[5]
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
        return spin2_energy(u_proj, mesh, params, spin_mats)
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

    u_out = SpinorState(copy(u_proj.u2), copy(u_proj.u1), copy(u_proj.u0), copy(u_proj.um1), copy(u_proj.um2))
    return τ_opt, f_opt, u_out
end


function get_new_energy(u::SpinorState, u_old::SpinorState, d::SpinorState, d_old::SpinorState, mesh, params, M, spin_mats, lower_bound::Float64, upper_bound::Float64, τ_fixed::Float64, step_control::String, retraction::String, ws::Workspace, metric)
    u_proj = ws.u_proj
    if step_control == "optimal"
            τ, E_new, u_new = optimal_tau_blackbox(u, d, u_proj, mesh, params, M, spin_mats, lower_bound, upper_bound, retraction)
    elseif step_control == "fixed"
        τ = τ_fixed
        get_retraction!(u_proj, u, d, τ, M, params.C_mag, retraction)
        E_new = spin2_energy(u_proj, mesh, params, spin_mats)
        u_new = u_proj
    elseif step_control == "MM"

       du_sq = sum(abs2, u.u2 .- u_old.u2) + 
                sum(abs2, u.u1 .- u_old.u1) + 
                sum(abs2, u.u0 .- u_old.u0) + 
                sum(abs2, u.um1 .- u_old.um1) + 
                sum(abs2, u.um2 .- u_old.um2)

        dg_sq = sum(abs2, d.u2 .- d_old.u2) +
                sum(abs2, d.u1 .- d_old.u1) + 
                sum(abs2, d.u0 .- d_old.u0) + 
                sum(abs2, d.um1 .- d_old.um1) + 
                sum(abs2, d.um2 .- d_old.um2)

            τ_prelim = sqrt(du_sq / dg_sq)
            τ = min(max(τ_prelim, lower_bound), upper_bound)
        get_retraction!(u_proj, u, d, τ, M, params.C_mag, retraction)

        E_new = spin2_energy(u_proj, mesh, params, spin_mats)
        u_new = SpinorState(copy(u_proj.u2), copy(u_proj.u1), copy(u_proj.u0), copy(u_proj.um1), copy(u_proj.um2))

    elseif step_control == "BB"
        du2 = u.u2 - u_old.u2
        du1 = u.u1 - u_old.u1 
        du0 = u.u0 - u_old.u0 
        dum1 = u.um1 - u_old.um1 
        dum2 = u.um2 - u_old.um2 

        du_vec = [real(du2); imag(du2) ; real(du1);imag(du1) ; real(du0);imag(du0) ; real(dum1);imag(dum1) ; real(dum2);imag(dum2)]
        du_vec = [du1; du0; dum1]

        dg2 = d.u2 - d_old.u2
        dg1 = d.u1 - d_old.u1 
        dg0 = d.u0 - d_old.u0 
        dgm1 = d.um1 - d_old.um1 
        dgm2 = d.um2 - d_old.um2
        dg_vec = [dg2; dg1; dg0; dgm1; dgm2]

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

        order = params.order_finite_element
        no_of_basis = (order == 1) ? 3 : 6

        phi_in_quad_point = MMatrix{6, 7, Float64}(undef)
        for qpt in 1:7
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

        get_retraction!(u_proj, u, d, τ, M, params.C_mag, retraction)
        E_new = spin2_energy(u_proj, mesh, params, spin_mats)
        u_new = SpinorState(copy(u_proj.u2), copy(u_proj.u1), copy(u_proj.u0), copy(u_proj.um1), copy(u_proj.um2))
    end

    return τ, E_new, u_new
end

function get_residual!(u, params, M, E_prime, ws)
    C_mag = params.C_mag
    Mu2  = ws.Mu2 
    Mu1  = ws.Mu1 
    Mu0  = ws.Mu0 
    Mum1 = ws.Mum1
    Mum2  = ws.Mum2 

    res_u2  = ws.res_u2
    res_u1  = ws.res_u1
    res_u0  = ws.res_u0
    res_um1 = ws.res_um1
    res_um2  = ws.res_um2

    u2 = u.u2
    u1  = u.u1
    u0  = u.u0
    um1 = u.um1
    um2 = u.um2

    mul!(Mu2,  M, u2)
    mul!(Mu1,  M, u1)
    mul!(Mu0,  M, u0)
    mul!(Mum1, M, um1)
    mul!(Mum2,  M, um2)


    b1 = real(dot(E_prime.u2, u2) + 
          dot(E_prime.u1, u1) + #<E'(u),u>
          dot(E_prime.u0, u0) +
          dot(E_prime.um1, um1) + 
          dot(E_prime.um2, um2)
          )

    b2 = real(2 * dot(E_prime.u2, u2) + dot(E_prime.u1, u1) - #<E'(u),Lu>
          dot(E_prime.um1, um1) - 2* dot(E_prime.um2, um2))
    LuLu = real(4 * dot(u2, Mu2) + dot(u1, Mu1) + dot(um1, Mum1) + 4 * dot(um2, Mum2))

    det_inv = 1.0 / (LuLu - C_mag^2)

    λ = det_inv * (LuLu * b1 - C_mag * b2)
    μ = det_inv * (b2 - C_mag * b1)

    res_u2  .= E_prime.u2  .- (λ + 2 * μ) .* Mu2
    res_u1  .= E_prime.u1  .- (λ + μ)     .* Mu1
    res_u0  .= E_prime.u0  .-  λ          .* Mu0
    res_um1 .= E_prime.um1 .- (λ- μ)     .* Mum1
    res_um2 .= E_prime.um2 .- (λ - 2 * μ) .* Mum2

    res = sqrt(real(
        dot(res_u1, M, res_u1) + dot(res_u0, M, res_u0) + dot(res_um1, M, res_um1)
    ))

    return res, λ, μ, SpinorState(Mu2, Mu1, Mu0, Mum1, Mum2)
end

function get_Riemannian_gradient(u, Mu, E_prime, mesh, params, metric, P, res, ws::Workspace)
    # ∇_A_N = A\Mu
    ws.MLu.u2 .= 2 * Mu.u2
    ws.MLu.u1 .= Mu.u1 
    ws.MLu.u0  .=  0
    ws.MLu.um1 .= -Mu.um1
    ws.MLu.um2 .= -2 * Mu.um2
    MLu = ws.MLu
    
    tol = 1e-3 * res #adaptive tolerance

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
        ws.∇_A_M.u2  .=  2 * ∇_A_N.u2
        ws.∇_A_M.u1  .=  ∇_A_N.u1
        ws.∇_A_M.u0  .=  0
        ws.∇_A_M.um1 .= -∇_A_N.um1
        ws.∇_A_M.um2 .= -2 * ∇_A_N.um2
        ∇_A_M = ws.∇_A_M
    end

    bN = real(
        dot(E_prime.u2, ∇_A_N.u2) +
        dot(E_prime.u1, ∇_A_N.u1) +
        dot(E_prime.u0, ∇_A_N.u0) +
        dot(E_prime.um1, ∇_A_N.um1) + 
        dot(E_prime.um2, ∇_A_N.um2)
    )

    bM = real(
        dot(E_prime.u2, ∇_A_M.u2) +
        dot(E_prime.u1, ∇_A_M.u1) +
        dot(E_prime.u0, ∇_A_M.u0) +
        dot(E_prime.um1, ∇_A_M.um1) + 
        dot(E_prime.um2, ∇_A_M.um2) 
    )

    NN = real(
        dot(Mu.u2, ∇_A_N.u2) +
        dot(Mu.u1, ∇_A_N.u1) +
        dot(Mu.u0, ∇_A_N.u0) +
        dot(Mu.um1, ∇_A_N.um1) + 
        dot(Mu.um2, ∇_A_N.um2)
    )

    NM = real(
        dot(Mu.u2, ∇_A_M.u2) +
        dot(Mu.u1, ∇_A_M.u1) +
        dot(Mu.u0, ∇_A_M.u0) +
        dot(Mu.um1, ∇_A_M.um1) + 
        dot(Mu.um2, ∇_A_M.um2) 
    )

    MN = real(
        dot(MLu.u2, ∇_A_N.u2) +
        dot(MLu.u1, ∇_A_N.u1) +
        dot(MLu.u0, ∇_A_N.u0) +
        dot(MLu.um1, ∇_A_N.um1) +
        dot(MLu.um2, ∇_A_N.um2)
    )

    MM = real(
        dot(MLu.u2, ∇_A_M.u2) +
        dot(MLu.u1, ∇_A_M.u1) +
        dot(MLu.u0, ∇_A_M.u0) +
        dot(MLu.um1, ∇_A_M.um1) +
        dot(MLu.um2, ∇_A_M.um2)
    )

    # Solve
    G = [
        NN NM
        MN MM
    ]

    rhs = [bN, bM]

    λ_A, μ_A  = G \ rhs

    grad = SpinorState(
        sobolev_grad.u2 - λ_A * ∇_A_N.u2 - μ_A  * ∇_A_M.u2,
        sobolev_grad.u1 - λ_A * ∇_A_N.u1 - μ_A  * ∇_A_M.u1,
        sobolev_grad.u0 - λ_A * ∇_A_N.u0 - μ_A  * ∇_A_M.u0,
        sobolev_grad.um1 - λ_A * ∇_A_N.um1 - μ_A * ∇_A_M.um1,
        sobolev_grad.um2 - λ_A * ∇_A_N.um2 - μ_A  * ∇_A_M.um2
    )

    return grad, λ_A, μ_A
 
end

function spin2_gradient_flow(
    u::SpinorState, mesh, params, M;
    metric::String = "spin",
    step_control::String = "optimal",
    retraction::String = "newton",
    fixed_τ::Float64 = 1.0
)
    @assert metric in ["spin", "spin-diagonal", "density", "A0"] "Invalid metric: $metric"
    @assert step_control in ["optimal", "fixed", "MM", "BB"] "Invalid step_control: $step_control"
    @assert retraction in ["newton", "inexact", "closed-form"] "Invalid retraction: $retraction"

    order_finite_element = params.order_finite_element
    Energies = Float64[]
    Residuals = Float64[]
    time_vec = Float64[]
    Diffs = Float64[]
    N2_vec = Float64[]
    N1_vec = Float64[]
    N0_vec = Float64[]
    Nm1_vec = Float64[]
    Nm2_vec = Float64[]
    taus = Float64[]

    spin_mats = SpinMatrices(params.F)
    E = spin2_energy(u, mesh, params, spin_mats)

    @printf("Initial energy: %.12f | Metric: %s | Step: %s | Retraction: %s\n", E, metric, step_control, retraction)
    push!(Energies, E)

    μ, λ = 0.0, 0.0
    res = 10.0
     dn_norm = 0.0
    residual_L2 = 10.0
    C_mag = params.C_mag  
    
    buf_m2 = similar(u.um2)
    buf_m1 = similar(u.um1)
    buf_0  = similar(u.u0)
    buf_1  = similar(u.u1)
    buf_2  = similar(u.u2)

    q = params.q 
    p = params.p

    dn = SpinorState(similar(u.u2), similar(u.u1), similar(u.u0), similar(u.um1), similar(u.um2))
    u_old      = SpinorState(similar(u.u2), similar(u.u1), similar(u.u0), similar(u.um1), similar(u.um2))
    dn_old = SpinorState(similar(u.u2), similar(u.u1), similar(u.u0), similar(u.um1), similar(u.um2))

    ws = Workspace(u, size(mesh.T,1))
    M, S, V = matrices(mesh, params)

    #Build preconditioner for CG
        P = lu(
                    2 * blockdiag(0.5 * S+ V + (4*q - 2*p) * M, 
                    0.5 * S+ V + (q - p) * M, 
                    0.5 * S + V, 
                    0.5 * S+ V + (q + p) * M,
                    0.5 * S+ V + (4*q + 2*p) * M)
                    )

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

        E_prime = frechet_derivative!(u, mesh, params, ws)
        res, λ, μ, Mu = get_residual!(u, params, M, E_prime, ws)
        push!(Residuals, res)

        gradR, λ_A, μ_A, =  get_Riemannian_gradient(u, Mu, E_prime, mesh, params, metric, P, res, ws)

        if j > 1
            copyto!(dn_old.u2,  dn.u2)
            copyto!(dn_old.u1,  dn.u1)
            copyto!(dn_old.u0,  dn.u0)
            copyto!(dn_old.um1, dn.um1)
            copyto!(dn_old.um2,  dn.um2)
        end

        #dn = -gradR
        dn.u2  .= -gradR.u2; dn.u1  .= -gradR.u1; dn.u0  .= -gradR.u0; dn.um1 .= -gradR.um1; dn.um2  .= -gradR.um2

        dn = SpinorState(copy(dn.u2), copy(dn.u1), copy(dn.u0), copy(dn.um1), copy(dn.um2))

        #check whether dn'*M*u = 0 = dn'*M*Lu
        mul!(buf_m2, M, dn.um2)
        mul!(buf_m1, M, dn.um1)
        mul!(buf_0,  M, dn.u0)
        mul!(buf_1,  M, dn.u1)
        mul!(buf_2, M, dn.u2)

        tangent_check_N = real(dot(u.um2, buf_m2)) + real(dot(u.um1, buf_m1) + dot(u.u0, buf_0) + dot(u.u1, buf_1)) + real(dot(u.u2, buf_2))
        tangent_check_M = real(-2.0 * dot(u.um2, buf_m2)-dot(u.um1, buf_m1) + dot(u.u1, buf_1) + 2.0 * dot(u.u2, buf_2))

        if abs(tangent_check_N) > 1e-8 || abs(tangent_check_M) > 1e-8
            @show(tangent_check_M, tangent_check_N)
        end

        if j <= 3
            if step_control == "MM" || step_control == "BB"
                 τ, E_new, u_new = get_new_energy(u, u_old, dn, dn_old, mesh, params, M, spin_mats, 1e-6, 1.0, fixed_τ, "optimal", retraction, ws, metric)
            elseif metric == "A0"
                τ, E_new, u_new = get_new_energy(u, u_old, dn, dn_old,  mesh, params, M, spin_mats, 1e-6, 1e-1, fixed_τ, step_control, retraction, ws, metric)
            else
                τ, E_new, u_new = get_new_energy(u, u_old, dn, dn_old,  mesh, params, M, spin_mats, 1e-6, 5.0, fixed_τ, step_control, retraction, ws, metric)
            end
        else
             τ, E_new, u_new = get_new_energy(u, u_old, dn, dn_old, mesh, params, M, spin_mats, lower_bound, upper_bound, fixed_τ, step_control, retraction, ws, metric)          
        end
        push!(taus, τ)
        push!(Energies, E_new)

        mul!(buf_m2, M, u_new.um2)
        mul!(buf_m1, M, u_new.um1)
        mul!(buf_0,  M, u_new.u0)
        mul!(buf_1,  M, u_new.u1)
        mul!(buf_2, M, u_new.u2)

        m_m2 = real(dot(u_new.um2, buf_m2))
        m_m1 = real(dot(u_new.um1, buf_m1))
        m_0  = real(dot(u_new.u0,  buf_0))
        m_1  = real(dot(u_new.u1,  buf_1))
        m_2 = real(dot(u_new.u2, buf_2))


        push!(N2_vec, m_2)
        push!(N1_vec, m_1)
        push!(N0_vec, m_0)
        push!(Nm1_vec, m_m1)
        push!(Nm2_vec, m_m2)

        sphere_check_N = m_m2 + m_m1 + m_0 + m_1 + m_2
        sphere_check_M = 2.0 * m_2 + m_1 - m_m1 - 2.0 * m_m2
        if retraction != "inexact"
            if abs(sphere_check_N - 1.0) > 1e-8 || abs(sphere_check_M - C_mag) > 1e-8
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

        copyto!(u_old.u2,  u.u2)
        copyto!(u_old.u1,  u.u1)
        copyto!(u_old.u0,  u.u0)
        copyto!(u_old.um1, u.um1)
        copyto!(u_old.um2, u.um2)
        
        u = SpinorState(copy(u_new.u2), copy(u_new.u1), copy(u_new.u0), copy(u_new.um1), copy(u_new.um2))

        iter_time = time() - t_start
  
        @printf("\n P%d [%s | %s | %s] Iter: %d | Energy: %.12f | λ: %.12f | μ: %.12f | diff: %.12e | Res: %.12e | τ: %.17e | Time: %.3f s\n", 
                order_finite_element, metric, step_control, retraction, j, E_new, λ/2, μ/2,  diff, res, τ, iter_time)
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

    Nm2 = real(dot(u.um2, M, u.um2))
    Nm1 = real(dot(u.um1, M, u.um1))
    N0  = real(dot(u.u0, M, u.u0))
    N1  = real(dot(u.u1, M, u.u1))
    N2  = real(dot(u.u2, M, u.u2))

    energy_status = energy_diminishing ?
    "The scheme was energy diminishing." :
    "The scheme was not energy diminishing."

    result_summary = @sprintf("""
        __________Final Result: [%s | %s | %s]__________
        P%d | Iter: %d | Energy: %.12f | λ: %.12f | μ: %.12f | Res: %.12e | Norm dn: %.12e | Average time per iteration: %.3f s | Total time: %.3f min
        Component masses: N₋₂ = %.12f | N₋₁ = %.12f | N₀ = %.12f | N₁ = %.12f | N₂ = %.12f
        %s
        """,
            metric, step_control, retraction, order_finite_element, length(Energies), Energies[end],
            λ / 2, μ / 2,  res, dn_norm, avg_time, total_time,
            Nm2, Nm1, N0, N1, N2, energy_status
        )

    return u, Energies, Residuals, Diffs, dn_norm, μ, λ, taus, N2_vec, N1_vec, N0_vec, Nm1_vec, Nm2_vec, time_vec, result_summary
end

function P2_reorder_coordinates(coords, coords_inner, u_approx)
    sorted_indices = sortperm(eachrow(coords), by=x -> (x[2], x[1])) #sort by y, then by x
    new_order = coords[sorted_indices, :]
    # bn_reordered = zeros(Int, size(new_order, 1), 1)
    # for i in axes(new_order, 1)
    #     if abs(new_order[i, 1]) == abs(a) || abs(new_order[i, 1]) == abs(a) == abs(b) || abs(new_order[i, 2]) == abs(a) || abs(new_order[i, 2]) == abs(a) == abs(b)
    #         bn_reordered[i] = 1
    #     end
    # end

    sorted_indices_inner = sortperm(eachrow(coords_inner), by=x -> (x[2], x[1]))
    um2 = u_approx.um2 
    um1 = u_approx.um1 
    u0 = u_approx.u0 
    u1 = u_approx.u1
    u2 = u_approx.u2

    um2_reordered = um2[sorted_indices_inner]
    um1_reordered = um1[sorted_indices_inner]
    u0_reordered = u0[sorted_indices_inner]
    u1_reordered = u1[sorted_indices_inner]
    u2_reordered = u2[sorted_indices_inner]

    u_reordered = SpinorState(u2_reordered, u1_reordered, u0_reordered, um1_reordered, um2_reordered)
    return  u_reordered
end

function plot_density_2D(u, mesh, params)

    um2 = u.um2
    um1 = u.um1 
    u0  = u.u0 
    u1  = u.u1 
    u2  = u.u2 

    Nd = mesh.x 
    a  = params.a 
    b  = params.b 
    N  = params.N 
    order_finite_element = params.order_finite_element

    if order_finite_element == 1
        l = div(length(um1), 2)
        Nx = Int(sqrt(size(Nd,1)) - 1)

        um2_grid = zeros(ComplexF64, Nx + 1, Nx + 1)
        um1_grid = zeros(ComplexF64, Nx + 1, Nx + 1)
        u0_grid  = zeros(ComplexF64, Nx + 1, Nx + 1)
        u1_grid  = zeros(ComplexF64, Nx + 1, Nx + 1)
        u2_grid  = zeros(ComplexF64, Nx + 1, Nx + 1)

        for j = 1:Nx - 1
            um2_grid[j+1, 2:end-1] = um2[(j-1)*(Nx-1) + 1 : (Nx-1)*j]
            um1_grid[j+1, 2:end-1] = um1[(j-1)*(Nx-1) + 1 : (Nx-1)*j]
            u0_grid[j+1, 2:end-1]  = u0[(j-1)*(Nx-1) + 1 : (Nx-1)*j]
            u1_grid[j+1, 2:end-1]  = u1[(j-1)*(Nx-1) + 1 : (Nx-1)*j]
            u2_grid[j+1, 2:end-1] = u2[(j-1)*(Nx-1) + 1 : (Nx-1)*j]
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

        um2 = u_reordered.um2
        um1 = u_reordered.um1
        u0  = u_reordered.u0
        u1 = u_reordered.u1
        u2 = u_reordered.u2

        Nx = Int(sqrt(size(Nd, 1)) - 1)
        l  = div(length(um1), 2)

        um2_grid = zeros(ComplexF64, Nx + 1, Nx + 1)
        um1_grid = zeros(ComplexF64, Nx + 1, Nx + 1)
        u0_grid  = zeros(ComplexF64, Nx + 1, Nx + 1)
        u1_grid  = zeros(ComplexF64, Nx + 1, Nx + 1)
        u2_grid  = zeros(ComplexF64, Nx + 1, Nx + 1)

        for j = 1:Nx - 1
            um2_grid[j+1, 2:end-1] = um2[(j-1)*(Nx-1) + 1 : (Nx-1)*j]
            um1_grid[j+1, 2:end-1] = um1[(j-1)*(Nx-1) + 1 : (Nx-1)*j]
            u0_grid[j+1, 2:end-1]  = u0[(j-1)*(Nx-1) + 1 : (Nx-1)*j]
            u1_grid[j+1, 2:end-1]  = u1[(j-1)*(Nx-1) + 1 : (Nx-1)*j]
            u2_grid[j+1, 2:end-1] = u2[(j-1)*(Nx-1) + 1 : (Nx-1)*j]
        end
    end


    # Density
    um2_density = abs2.(um2_grid)
    um1_density = abs2.(um1_grid)
    u0_density  = abs2.(u0_grid)
    u1_density  = abs2.(u1_grid)
    u2_density = abs2.(u2_grid)


    # Colormap and Shared Color Scale
    cmin = min(
        minimum(um2_density),
        minimum(um1_density),
        minimum(u0_density),
        minimum(u1_density),
        minimum(u2_density),
    )

    cmax = max(
        maximum(um2_density),
        maximum(um1_density),
        maximum(u0_density),
        maximum(u1_density),
        maximum(u2_density),
    ) * 1.05

    crange_density = (cmin, cmax)
    cmap_density = :turbo


fig = CairoMakie.Figure(size = (2500, 550))

density_scale = x -> x^0.7

ticks_size = 28

ax1 = CairoMakie.Axis(
    fig[1, 1],
    title = "|u₂|²" ,
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
    ax1, x_range, y_range, u2_density;
    colormap = cmap_density,
    colorrange = crange_density,
    colorscale = density_scale
)

ax2 = CairoMakie.Axis(
    fig[1, 2],
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

 CairoMakie.heatmap!(
    ax2, x_range, y_range, u1_density;
    colormap = cmap_density,
    colorrange = crange_density,
    colorscale = density_scale
)

ax3 = CairoMakie.Axis(
    fig[1, 3],
    title = "|u₀|²" ,
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
    ax3, x_range, y_range, u0_density;
    colormap = cmap_density,
    colorrange = crange_density,
    colorscale = density_scale
)

ax4 = CairoMakie.Axis(
    fig[1, 4],
    title = "|u₋₁|²" ,
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
    ax4, x_range, y_range, um1_density;
    colormap = cmap_density,
    colorrange = crange_density,
    colorscale = density_scale
)

ax5 = CairoMakie.Axis(
    fig[1, 5],
    title = "|u₋₂|²" ,
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
    ax5, x_range, y_range, um2_density;
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
    fig[1, 6],
    h1;
    # label = "Density",
    width = 18,
    height = 370,
    labelpadding = 15,
    ticks = density_ticks,
    ticklabelsize = ticks_size,
    tickformat = values -> [@sprintf("%.3f", v) for v in values]
)

colgap!(fig.layout, 45)

display(fig)

    return X, Y, um2_density, um1_density, u0_density, u1_density, u2_density
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
    β0 = 183.9
    β1 = 26.8
    β2 = 134.7
# Linear Zeeman parameter
    p = 0.0
# Quadratic Zeeman parameter
    q = 0.5
# Spin-F
    F = 2
# Specify potential
    V_fun(x) =  0.5 * (x[1]^2 + x[2]^2) + 25 * sin(pi*x[1]/2)^2 + 25 * sin(pi*x[2]/2)^2
# Specify magnetization constant
    C_mag = 0.3 
# Specify maximum number of iterations
    maxIter = 6000
# Stopping criterion: Res < tol
tol = 1e-8

# Initialize mesh and parameter structs
    params = Params(order_finite_element, a, b, F, N, β0, β1, β2, p, q, V_fun, C_mag, maxIter, tol)
    mesh = fem2d_uniform_mesh(params)
# build mass, stiffness and potential matrix ONCE
    M, S, V = matrices(mesh, params)

# Initial value
    ζ(x) = pi^(-1/2) * exp(-0.5 * (x[1]^2 + x[2]^2))
    init_phi = initial_phi(mesh, ζ, M)
    u_init= SpinorState(sqrt(0.0077) * init_phi, sqrt(0.3001) * init_phi, sqrt(0.6769)*init_phi, sqrt(0.0152) * init_phi, sqrt(0.0002) * init_phi)

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
        # "inexact"
        # "closed-form"

    (u_final, Energies, Residuals, Diffs, dn_norm, μ, λ, taus, N1_vec, N0_vec, Nm1_vec, time_vec, result_summary) = spin2_gradient_flow(
                u_init, mesh, params, M;
                metric = "A0",
                step_control = "MM",
                retraction = "newton"
            )
# Print final results
    println(result_summary)

# Plot density
    plot_density_2D(u_final, mesh, params)

