library(plyr)
library(fGarch)
library(corrplot)

setwd("G:/My Drive/M6 submission platform/GitHub")
load("Score compute.Rdata")

## Hypothesis No.2 ----
#There will be a small group of participants that clearly outperform the average both 
#in terms of forecast accuracy and portfolio returns. 

#Fetch teams included in the Global leaderboard
sg <- score_summary[score_summary$pid=="Global",]
#Performance of the benchmark
bRPS <- sg[sg$sid=="32cdcc24",]$RPS
bIR <- sg[sg$sid=="32cdcc24",]$IR
sg <- sg[sg$sid!="32cdcc24",]
#Average performance - benchmark excluded
mRPS <- mean(sg$RPS) #mRPS <- median(sg$RPS)
mIR <- mean(sg$IR) # mIR <- median(sg$IR)

#par(mfrow=c(2,2))
examine_m <- sg[sg$RPS<mRPS & sg$IR>mIR,]
plot(density(examine_m$IR-mIR, from=0), xlab="", main="") #IR difference - Average
plot(density(mRPS-examine_m$RPS, from=0), xlab="", main="") #RPS difference - Average

examine_b <- sg[sg$RPS<bRPS & sg$IR>bIR,]
plot(density(examine_b$IR-bIR, from=0), xlab="", main="") #IR difference - Benchmark
plot(density(bRPS-examine_b$RPS, from=0), xlab="", main="") #RPS difference - Benchmark

#Statistics
nrow(examine_m)
nrow(examine_m)-nrow(examine_m[examine_m$Returns>0,])
summary(mRPS-examine_m$RPS)/mRPS
nrow(examine_b)
summary(examine_b$IR-bIR)/bIR
summary(bRPS-examine_b$RPS)/bRPS