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


df = CSV.read("C:\\Users\\Dusan Tomic\\Desktop\\Fakultet\\II Godina\\primenjeni algoritmi\\K2 priprema\\projekat alg\\procesor.csv", DataFrame)
println("DataSet pre obrade:")
println()
display(describe(df))

display(countmap(df[!, :coreNum]))
df[ismissing.(df[!, :coreNum]), :coreNum] .= mode(skipmissing(df[!, :coreNum]))

display(countmap(df[!, :threadNum]))
df[ismissing.(df[!, :threadNum]), :threadNum] .= mode(skipmissing(df[!, :threadNum]))

display(countmap(df[!, :frekvencija]))
df[ismissing.(df[!, :frekvencija]), :frekvencija] .= mode(skipmissing(df[!, :frekvencija]))

display(countmap(df[!, :RAMspeed]))
df[ismissing.(df[!, :RAMspeed]), :RAMspeed] .= mode(skipmissing(df[!, :RAMspeed]))

display(countmap(df[!, :DDRTip]))
df[ismissing.(df[!, :DDRTip]), :DDRTip] .= mode(skipmissing(df[!, :DDRTip]))

println("DataSet posle obrade:")
println()
display(describe(df))

data_train, data_test = TrainTestSplit(df, 0.8) #splitovanje podataka

nezavisne_train = convert(Matrix{Float64},select(data_train, Not([:jacina])))'
zavisne_train = convert(Matrix{Float64},select(data_train, :jacina))'

nezavisne_test = convert(Matrix{Float64},select(data_test, Not([:jacina])))'
zavisne_test = convert(Matrix{Float64},select(data_test, :jacina))'

#ovde ili relu ili σ
model = Dense(5, 1, σ)

#=model = Chain(
    Dense(3, 2, σ),
    Dense(2, 1),
    softmax
)=#

# x-> inputi
# y=> prediction
loss(x, y) = Flux.binarycrossentropy(model(x), y)
#=koristimo crossentropy tip za loss fju, koristi se kad imamo multi-klasifikaciju
  mozda je ovo potencijalni problem=#

parametri = Flux.params(model)
#= params() fja kreira objekat parametara koji upucuje na trainable parametre
   ovaj objekat je neophodan za proces trenaze nad podacima=#

learningRate = 0.01
optimizer = Flux.Optimise.Adam(learningRate)

loss_history = []
trainData = [(nezavisne_train ,zavisne_train)]

ciklusi = 1200

for i in 1:ciklusi
    
    Flux.train!(loss,parametri,trainData,optimizer)
    train_loss = loss(nezavisne_train,zavisne_train)
    
    push!(loss_history,train_loss)
    #println("Ciklus $i : Training Loss = $train_loss")

end

predikcija = model(nezavisne_test)
println(predikcija)
#length(predikcija)

dataPredictedTestClass = repeat(0:0, length(predikcija)) #da se pretovori u integere
for j=1:length(predikcija)
  if (predikcija[j] <0.5)
        dataPredictedTestClass[j] = 0
    else
        dataPredictedTestClass[j] = 1
    end
end

println("Predvidjena jacina: $dataPredictedTestClass)")
println("Jacina iz zadata iz csv-a: $(data_test.jacina))")

FPT = 0 # false positives
FNT = 0 # false negatives
TPT = 0 # true positives
TNT = 0 # true negatives
for k in 1:length(predikcija)
    if data_test.jacina[k] == 0 && predikcija[k] == 0
        global TNT += 1;
    elseif data_test.jacina[k] == 0 && predikcija[k] == 1
        global FPT +=1
    elseif data_test.jacina[k] == 1 && predikcija[k] == 0
        global FNT +=1
    elseif data_test.jacina[k] == 1 && predikcija[k] == 1
        global TPT +=1
    end
end

preciznost = (TPT+TNT)/(TPT+TNT+FPT+FNT)
osetljivost = TPT/(TPT+FNT)
specificnost = TNT/(TNT+FPT)

println("TP = $TPT, FP = $FPT, TN =$TNT, FN = $FNT")
println("Preciznost za test skup je $preciznost")
println("Osetljivost za test skup je $osetljivost")
println("Specificnost za test skup je $specificnost")

rocTest = ROC.roc(predikcija, data_test.jacina, true)
aucTest = AUC(rocTest)
println("Povrsina ispod krive u procentima je: $aucTest")

if (aucTest>0.9)
    println("Klasifikator je jako dobar")
elseif (aucTest>0.8)
    println("Klasifikator je veoma dobar")
elseif (aucTest>0.7)
    println("Klasifikator je dosta dobar")
elseif (aucTest>0.5)
    println("Klasifikator je relativno dobar")
else
    println("Klasifikator je los")
end

plot(rocTest, label="ROC kriva")
    




