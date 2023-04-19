#Autor: Yuliia Melnyk

include("blocksys.jl")

using Test
using .blocksys
using LinearAlgebra

sizes = [16, 10000,50000,100000,300000,500000]


@testset "Tests for matrix $size" for size in sizes      
    x = ones(size)

    println("Size: ", size)

    @testset "Gauss elimination" begin
        A,n,l = readMatrix("./$(size)/A.txt")
        b = readVector("./$(size)/b.txt")

        dx = gaussEliminationSolver(A,n,l,b)
        println(norm(x - dx)/norm(x))
        @test isapprox(dx, x)
    end

    @testset "Gauss elimination with partial pivot" begin
        A,n,l = readMatrix("./$(size)/A.txt")
        b = calculateRightSideVector(A,n,l)
        dx = gaussEliminationWithPartialPivotSolver(A,n,l,b)
        @test isapprox(dx, x)
    end

    @testset "LU decomposition" begin
        A,n,l = readMatrix("./$(size)/A.txt")
        b = calculateRightSideVector(A,n,l)
        dx = luSolver(A,n,l,b)
        @test isapprox(dx, x)
    end

    @testset "LU decomposition with partial pivot" begin
        A,n,l = readMatrix("./$(size)/A.txt")
        b = calculateRightSideVector(A,n,l)
        dx = luWithPartialPivotSolver(A,n,l,b)
        @test isapprox(dx,x)
    end

    
end

