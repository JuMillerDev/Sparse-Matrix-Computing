#Autor: Yuliia Melnyk

include("blocksys.jl")
#include("matrixgen.jl")
#using .matrixgen
using .blocksys
using LinearAlgebra

function benchmark(l, repeats, func, n)
    totalTime = 0
    totalMemory = 0
    for i in 1:repeats
        #matrixgen.blockmat(n, l, 10.0, "test.txt")
        A = readMatrix("./$(n)/A.txt")
        b = readVector("./$(n)/b.txt")
        (_, time, memory) = @timed func(A[1], n, l, b)
        x = func(A[1], n, l, b)
        totalTime += time
        totalMemory += memory
    end
    println(n, "; ",totalTime/repeats, "; ", totalMemory/repeats)
end

function classic(A,n,l,b)
    return \(Array(A),b)
end

functions = [#classic,
            gaussEliminationSolver, gaussEliminationWithPartialPivotSolver, luSolver, luWithPartialPivotSolver]

sizes = [#16,10000,50000,100000,300000,
    500000]


for j in eachindex(functions)
    println("Algorytm $j")
        
    #for k in eachindex(sizes)
        benchmark(4, 1, functions[j], 500000)
    #end
end 

testAndWrite()

