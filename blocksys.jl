#Autor: Yuliia Melnyk

module blocksys

import SparseArrays
using LinearAlgebra

export readMatrix, readVector, writeVector, writeVectorWithRelativeError,calculateRightSideVector, gaussElimination, gaussEliminationSolver, gaussEliminationWithPartialPivot, gaussEliminationWithPartialPivotSolver, lu, luSolver, luWithPartialPivot, luWithPartialPivotSolver, testAndWrite
"""
Funkcja wczytuje macierz z pliku
Dane wejściowe:
    file - ścieżka pliku wejściowego
Dane wyjściowe:
    (A, n, l), gdzie:
        A - macierz rzadka w formacie SparseMatrixCSC
        n - rozmiar macierzy
        l - rozmiar bloku
"""

function readMatrix(file::String)
    open(file) do f
        line = split(readline(f))
        n = parse(Int64, line[1])
        l = parse(Int64, line[2])

        rows = Int32[]
        cols = Int32[]
        vals = Float64[]

        for line in eachline(f)
            line = split(line)
            push!(rows, parse(Int32, line[1]))
            push!(cols, parse(Int32, line[2]))
            push!(vals, parse(Float64, line[3]))
        end

        for i in 2:(n/l)

            i1_C = (i-2)*(l)
            j1_C = (i-1)*(l)

            #fill C
            for m in 1:(l)
                for n in m:(l)
                    push!(rows, i1_C+n)
                    push!(cols, j1_C+m)
                    push!(vals, 0.0)
                end
            end
        end

        #fill pivot checkings
        for m in 1:n
            for n in (m+l)+1 :min(m+l+l,n)
                push!(rows, m)
                push!(cols, n)
                push!(vals, 0.0)
            end
        end

        A = SparseArrays.sparse(rows, cols, vals)
        println("Macierz A: ", A)
        return (A, n, l)
    end
end

"""
Funkcja wczytująca wektor z pliku
Dane wejściowe:
    file - ścieżka pliku wejściowego
Dane wyjściowe:
    vector - wczytany wektor
"""

function readVector(file::String)
    open(file) do f
        vector = Float64[]

        #skip line with size
        readline(f)

        for line in eachline(f)
            push!(vector, parse(Float64, line))
        end

        return vector
    end
end

"""
Funkcja zapisująca wektor do pliku
Dane wejściowe:
    file - ścieżka do pliku
    x - wektor, który chcemy zapisać
"""

function writeVector(file::String, x::Vector)
    open(file, "w") do f
        foreach(line -> println(f, string(line), "\n"), x)
    end
end

"""Funkcja zapisująca wektor do pliku wraz z błędem względnym
Dane wejściowe:
    file - ścieżka do pliku
    x - wektor, który chcemy zapisać
    A - macierz rzadka w formacie SparseMatrixCSC
    b - wektor prawych stron
"""

function writeVectorWithRelativeError(file::String, x::Vector)
    open(file, "w") do f
        y = ones(Float64, length(x))
        write(f, string(norm(y - x) / norm(x)), "\n")
        foreach(a->write(f, string(a), "\n"), x)
    end
end

"""
Funkcja wyliczająca wektor prawych stron na podstawie macierzy A
Dane wejściowe:
    A - macierz rzadka w formacie SparseMatrixCSC
    n - rozmiar macierzy
    l - rozmiar bloku
Dane wyjściowe:
    b - wektor prawych stron
"""

function calculateRightSideVector(A::SparseArrays.SparseMatrixCSC, n::Int64, l::Int64)
    #x = ones(Float64, n) #zakladamy, ze x = 1
    b = zeros(Float64, n)

    for i in 1:n
        for j in max(1,(i-(i%l))-l):min(i + l, n)
            b[i] += A[i, j]
        end
    end

    return b
end

"""
Funkcja rozwiązująca układ równań Ax = b metodą eliminacji Gaussa
Dane wejściowe:
    A - macierz rzadka współczynników w formacie SparseMatrixCSC
    n - rozmiar macierzy
    l - rozmiar bloku
    b - wektor prawych stron
Dane wyjściowe:
    x - wektor wyników
"""

function gaussEliminationSolver(A::SparseArrays.SparseMatrixCSC, n::Int64, l::Int64, b::Vector{Float64})::Vector{Float64}

    #eliminacja Gaussa
    for k in 1:(n-1)
        bound = min(((k+l)-(k%l))+1,n)
        for i in (k+1):bound
            z = A[i, k] / A[k, k]
            A[i, k] = 0.0
            for j in (k+1):min(k+l, n)
                A[i, j] -= z * A[k, j]
            end

            b[i] -= z * b[k]
        end
    end

    #rozwiazanie ukladu
    x = zeros(n)

    for i in n:-1:1
        x[i] = b[i]
        for j in (i+1):min(i+l, n)
            x[i] -= A[i, j] * x[j]
        end
        x[i] /= A[i, i]
    end

    return x
end

"""
Funkcja rozwiązująca układ równań Ax = b metodą eliminacji Gaussa (z częściowym wyborem elementu głównego)
Dane wejściowe:
    A - macierz rzadka współczynników w formacie SparseMatrixCSC
    n - rozmiar macierzy
    l - rozmiar bloku
    b - wektor prawych stron
Dane wyjściowe:
    x - wektor wyników
"""

function gaussEliminationWithPartialPivotSolver(A::SparseArrays.SparseMatrixCSC, n::Int64, l::Int64, b::Vector{Float64})::Vector{Float64}
    #eliminacja Gaussa
    perm = [1:n;]

    #println("Permutacja początkowa: ", perm)

    for k in 1:(n-1)
        max = abs(A[perm[k], k])
        maxIndex = k

        bound = min(((k+l)-(k%l))+1, n) #buttom row

        for i in k:bound
            if abs(A[perm[i], k]) > max
                max = abs(A[perm[i], k])
                maxIndex = i
            end
        end

        perm[maxIndex], perm[k] = perm[k], perm[maxIndex]

        for i in (k+1):bound
            z = A[perm[i], k] / A[perm[k], k]
            A[perm[i], k] = 0.0
            # p[k] to najwyżej k + block_size, ponieważ poniżej są już zera
            for j in (k+1):min(k+(2*l), n)
                A[perm[i], j] -= z * A[perm[k], j]
            end
            b[perm[i]] -= z * b[perm[k]]
        end
    end

    #rozwiazanie ukladu
    x = zeros(Float64, n)

    for i in n:-1:1
        x[i] = b[perm[i]]
        for j in (i+1):min(i+(2*l), n)
            x[i] -= A[perm[i], j] * x[j]
        end
        x[i] /= A[perm[i], i]
    end

    return x
end

"""
Funkcja rozwiązująca układ równań Ax = b korzystając z rozkładu LU
Dane wejściowe:
    LU - macierz w formacie SparseMatrixCSC
    n - rozmiar macierzy
    l - rozmiar bloku
    b - wektor prawych stron
Dane wyjściowe:
    x - wektor wyników
"""

function luSolver(LU::SparseArrays.SparseMatrixCSC, n::Int64, l::Int64, b::Vector{Float64})::Vector{Float64}
    # A = L * U
    #eliminacja LU

    for k in 1:(n-1)
        bound = min(((k+l)-(k%l))+1, n)
        for i in (k+1):bound
            z = LU[i, k] / LU[k, k]
            LU[i, k] = z
    
            for j in (k+1):min(k+l, n)
                LU[i, j] -= z * LU[k, j]
            end
        end
    end

    #rozwiazanie ukladu

    for k in 1:(n-1)
        bound = min(((k+l)-(k%l))+1, n)
        for i in (k+1):bound
            b[i] -= LU[i, k] * b[k]
        end
    end

    x = zeros(Float64, n)

    #do aktualizacji elementów macierzy odwracam kolejność
    for i in n:-1:1
        x[i] = b[i]
        for j in (i+1):min(i+l, n)
            x[i] -= LU[i, j] * x[j]
        end
        x[i] /= LU[i, i]
    end

    return x
end

"""
Funkcja rozwiązująca układ równań Ax = b korzystając z rozkładu LU (z częściowym wyborem elementu głównego)
Dane wejściowe:
    A - macierz rozkładu LU w formacie SparseMatrixCSC 
    n - rozmiar macierzy
    l - rozmiar bloku
    b - wektor prawych stron
    perm - wektor permutacji
Dane wyjściowe:
    x - wektor wyników
"""

function luWithPartialPivotSolver(LU::SparseArrays.SparseMatrixCSC, n::Int64, l::Int64, b::Vector{Float64})::Vector{Float64}

    # A = L * U
    #eliminacja LU
    
    perm = [1:n;]

    for k in 1:(n-1)
        max = abs(LU[perm[k], k])
        maxIndex = k

        bound = min((k+l-(k%l))+1, n)

        for i in k+1:bound
            if abs(LU[perm[i], k]) > max
                max = abs(LU[perm[i], k])
                maxIndex = i
            end
        end

        perm[maxIndex], perm[k] = perm[k], perm[maxIndex]

        for i in (k+1):bound
            z = LU[perm[i], k] / LU[perm[k], k]

            LU[perm[i], k] = z

            for j in (k+1):min(k+(2*l), n)
                LU[perm[i], j] -= z * LU[perm[k], j]
            end
        end
    end


    #LZ = PB
    for k in 1:(n-1)
        for i in k+1:min(k+(2*l), n)
            b[perm[i]] -= LU[perm[i], k] * b[perm[k]]
        end
    end

    #UX = Z
    x = zeros(Float64, n)

    for i in n:-1:1
        x[i] = b[perm[i]]
        for j in (i+1):min(i+(2*l), n)
            x[i] -= LU[perm[i], j] * x[j]
        end
        x[i] /= LU[perm[i], i]
    end

    return x
end

end