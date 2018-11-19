module FiniteDifferences

using LinearAlgebra
using SparseArrays

import Base: eltype, show

struct Basis{T}
    j::Base.OneTo
    ρ::T
    Z::T
    δβ₁::T # Correction used for bare Coulomb potentials, Eq. (22) Schafer2009
end

Basis(n::I, ρ::T, Z::T=one(T)) where {I<:Integer, T} =
    Basis{T}(Base.OneTo(n), ρ, Z, Z*ρ/8 * (one(T) + Z*ρ))

# Basis(rₘₐₓ::T, n::I, Z::T=one(T)) where {I<:Integer,T} =
#     Basis{T}(n, rₘₐₓ/(n-1/2), Z)

eltype(::Basis{T}) where T = T

locs(basis::Basis{T}) where T = (basis.j .- 1/2)*basis.ρ

function (basis::Basis)(D::Diagonal)
    n = length(basis.j)
    @assert size(D) == (n,n)
    D
end
(basis::Basis)(f::Function) = Diagonal(f.(locs(basis)))
(basis::Basis)(::UniformScaling) = I

# Not really useful, only implemented to be compatible with FEDVR and
# BSplines.
function (basis::Basis)(r::AbstractVector)
    B = spzeros(eltype(r), length(r), length(basis.j))
    for j = basis.j[1:end-1]
        a = (j-1/2) * basis.ρ
        b = (j+1/2) * basis.ρ
        B[:,j] = (a .<= r .< b)
    end
    B
end

# Variationally derived coefficients for three-point stencil
α(j::Integer) = j^2/(j^2 - 1/4)
β(j::Integer) = (j^2 - j + 1/2)/(j^2 - j + 1/4)

function β(basis::Basis)
    b = β.(basis.j)
    b[1] += basis.δβ₁
    b
end

function derop(::Type{U}, basis::Basis{T}, o::Integer) where {U,T}
    o ∉ 0:2 && error("Unsupported derivative order $(o) ∉ 0:2")
    if o == 0
        basis(I)
    elseif o == 1
        a = α.(basis.j[1:end-1])
        Tridiagonal{U}(-a, zeros(T, length(basis.j)), a)/2basis.ρ
    else
        SymTridiagonal{U}(-2β(basis), α.(basis.j[1:end-1]))/basis.ρ^2
    end        
end
derop(basis::Basis{T}, o::Integer) where T = derop(T, basis, o)

show(io::IO, basis::Basis) = 
    write(io, "Finite differences basis with $(length(basis.j)) points spaced by ρ = $(basis.ρ)")

end # module
