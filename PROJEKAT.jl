#pr27/2021 Milan Tripkovic

using Flux
using DataFrames
using CSV
using StatsBase
using Lathe.preprocess: TrainTestSplit
using Plots, StatsPlots 
using Statistics
using StatsModels
using LinearAlgebra
using ROC


df = CSV.read("D:\\1\\podaci.csv", DataFrame)
println("DataSet :")
println()
display(describe(df))
#dopuna podataka th izbacivanje praznih
display(countmap(df[!, :idigraca]))
df[ismissing.(df[!, :idigraca]), :idigraca] .= mode(skipmissing(df[!, :idigraca]))

display(countmap(df[!, :brmeceva]))
df[ismissing.(df[!, :brmeceva]), :brmeceva] .= mode(skipmissing(df[!, :brmeceva]))

display(countmap(df[!, :brpobeda]))
df[ismissing.(df[!, :brpobeda]), :brpobeda] .= mode(skipmissing(df[!, :brpobeda]))

display(countmap(df[!, :brkilova]))
df[ismissing.(df[!, :brkilova]), :brkilova] .= mode(skipmissing(df[!, :brkilova]))

display(countmap(df[!, :brasista]))
df[ismissing.(df[!, :brasista]), :brasista] .= mode(skipmissing(df[!, :brasista]))

println("DataSet posle obrade:")
println()
display(describe(df))

data_train, data_test = TrainTestSplit(df, 0.8) #splitovanje podataka

nezavisne_train = convert(Matrix{Float64},select(data_train, Not([:toxican])))'#pravi matricu
zavisne_train = convert(Matrix{Float64},select(data_train, :toxican))'

nezavisne_test = convert(Matrix{Float64},select(data_test, Not([:toxican])))'
zavisne_test = convert(Matrix{Float64},select(data_test, :toxican))'


model = Dense(5, 1, σ)#5 inputa 1 output kao interfejs
# Dense postavlja početne vrednosti slobonih članova kao slučajne,kljuc je 0

loss(x, y) = Flux.binarycrossentropy(model(x), y)#x input y predikt


parametri = Flux.params(model)#kreira objekat parametara koji upucuje na trainable parametre
 

stopaucenja = 0.01
optimizer = Flux.Optimise.Adam(stopaucenja)#adaptive moment estimation

loss_history = []
trainData = [(nezavisne_train ,zavisne_train)]

ciklusi = 1200

for i in 1:ciklusi
    
    Flux.train!(loss,parametri,trainData,optimizer)
    train_loss = loss(nezavisne_train,zavisne_train)
    
    push!(loss_history,train_loss)#jede procesor


end

predikcija = model(nezavisne_test)
println(predikcija)


dataPredictedTestClass = repeat(0:0, length(predikcija)) #da se pretovori u integere
for j=1:length(predikcija)
  if (predikcija[j] <0.5)
        dataPredictedTestClass[j] = 0
    else
        dataPredictedTestClass[j] = 1
    end
end

println("Predvidjena toxicnosti: $dataPredictedTestClass)")
println("toxicnost iz zadata iz csv-a: $(data_test.toxican))")
#testiranje
FPT = 0 # false positives
FNT = 0 # false negatives
TPT = 0 # true positives
TNT = 0 # true negatives
for k in 1:length(predikcija)
    if data_test.toxican[k] == 0 && predikcija[k] == 0
        global TNT += 1;
    elseif data_test.toxican[k] == 0 && predikcija[k] == 1
        global FPT +=1
    elseif data_test.toxican[k] == 1 && predikcija[k] == 0
        global FNT +=1
    elseif data_test.toxican[k] == 1 && predikcija[k] == 1
        global TPT +=1
    end
end

preciznost = (TPT+TNT)/(TPT+TNT+FPT+FNT)
osetljivost = TPT/(TPT+FNT)
specificnost = TNT/(TNT+FPT)

println("TP = $TPT, FP = $FPT, TN =$TNT, FN = $FNT")
println("Preciznost  je $preciznost")
println("Osetljivost  je $osetljivost")
println("Specificnost  je $specificnost")

rocTest = ROC.roc(predikcija, data_test.toxican, true)
aucTest = AUC(rocTest)
println("Povrsina je: $aucTest")

if (aucTest>0.9)
    println(" jako dobar")
elseif (aucTest>0.8)
    println("veoma dobar")
elseif (aucTest>0.7)
    println(" dosta dobar")
elseif (aucTest>0.5)
    println(" relativno dobar")
else
    println("  los je")
end

plot(rocTest, label="ROC kriva")
    




