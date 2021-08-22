#################################################
# This code is taken from Alaettin Zubaroğlu    #
# Edited some parts for adapting ESTRA          #
#################################################

library(rJava)
library(proxy)
library(mclust)
library(streamMOA)
library(stream)
library(funtimes) # for purity

args<-commandArgs(TRUE)

xfname = args[1] # data except labels
lfname = args[2] # labels
k = as.numeric(args[3])
window_length = as.numeric( args[4])
m = as.numeric(args[5])
part_size =  1000# data process size
data_length =  as.numeric(args[6])

part_start_indexes = seq(1, (data_length-part_size+1), by=part_size)

Sys.time()
# cat("New run of CluStream algorithm PART by PART at :")
# cat("window length (horizon) : [", window_length, "]\n")

X = read.table(xfname, sep=",")
lbls = read.table(lfname)
lbls = t(lbls)
streammem = DSD_Memory(x=X, class=lbls, description="memo desc", loop=TRUE)
clustream = DSC_CluStream(m=m, horizon=window_length, k=k)

all_assign = c()
reset_stream(streammem, pos = 1)
total_time = 0
aris = c()
purr = c()

for(si in part_start_indexes)
{
    begin = Sys.time()
    update(clustream, streammem, part_size)
    
    assign = get_assignment(clustream, tail(head(X, si+part_size-1), part_size), type = "macro")
    assign[is.na(assign)] = -1
   
    end = Sys.time()
    this_time = end - begin
    total_time = total_time + this_time
    real_labels = tail(head(t(lbls), si+part_size-1), part_size)
    # cat("Found Labels: " , assign, "\n")
    # cat("Real Labels : " , real_labels, "\n")
    ari = adjustedRandIndex(assign, real_labels)
    pur = purity(real_labels, assign)
    pur = pur[[1]]
    #cat("Purity : " , pur, "\n")
    purr = c(purr,pur)
    all_assign = c(all_assign, assign)
    aris = c(aris, ari)
    cat("<ACCURACY_START>",si, ":", si+part_size-1, "datalength:", data_length, "acc", ari, "pur", pur, "meanpur", mean(na.omit(purr)), "meanacc", mean(na.omit(aris)), "time", total_time,"<ACCURACY_END>\n")
}

#cat("Total Time of this stream : [", total_time, "] seconds, average ari : [", mean(na.omit(aris)), "]\n")
#total_ari = adjustedRandIndex(all_assign, head(t(lbls), length(all_assign)))
#cat("Total ari : [", total_ari, "]\n")
