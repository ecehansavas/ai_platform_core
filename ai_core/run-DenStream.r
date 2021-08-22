#################################################
# This code is taken from Alaettin Zubaroğlu    #
# Edited some parts for adapting ESTRA          #
#################################################
library(rJava)
library(proxy)
library(stream)
library(streamMOA)
library(mclust)
library(funtimes) # for purity

args<-commandArgs(TRUE)

xfname = args[1]
lfname = args[2]
k = as.numeric(args[3])
epsilon = as.numeric(args[4])
beta = as.numeric(args[5])
part_size =  1000 # data process size
data_length = as.numeric(args[6])

Sys.time()
part_start_indexes = seq(1, (data_length-part_size+1), by=part_size)

X = read.table(xfname, sep=",")
lbls = read.table(lfname)
lbls = t(lbls)
streammem = DSD_Memory(x=X, class=lbls, description="memo desc", loop=TRUE)
denstream = DSC_DenStream( epsilon=epsilon, beta=beta, k=k)

#denstream = DSC_DenStream(initPoints = 1000, epsilon=epsilon, offline=20, lambda = 0.25,beta =beta, mu=10, k=k)


all_found_labels = c()
reset_stream(streammem, pos = 1)
total_time = 0
aris = c()
purr = c()

for(si in part_start_indexes)
{
    begin = Sys.time()
    update(denstream, streammem, part_size)
    
    found_label = get_assignment(denstream, tail(head(X, si+part_size-1), part_size), type = "macro")
    found_label[is.na(found_label)] = -1
    end = Sys.time()
    this_time = end - begin
    total_time = total_time + this_time
    real_labels = tail(head(t(lbls), si+part_size-1), part_size)
    # cat("Found Labels: " , found_label, "\n")
    # cat("Real Labels : " , real_labels, "\n")
    ari = adjustedRandIndex(found_label, real_labels)
    pur = purity(real_labels, found_label)
    pur = pur[[1]]
    # cat("Purity : " , pur, "\n")
    purr = c(purr,pur)
    all_found_labels = c(all_found_labels, found_label)
    aris = c(aris, ari)
    # cat("Indexes : [", si, ":", si+part_size-1, " ] ari : [", ari, "] Execution Time : [", this_time, "] seconds.\n")
    cat("<ACCURACY_START>",si, ":", si+part_size-1, "datalength:", data_length, "acc", ari, "pur", pur, "meanpur", mean(na.omit(purr)), "meanacc", mean(na.omit(aris)), "time", total_time,"<ACCURACY_END>\n")
}

# cat("Total Time of this stream : [", total_time, "] seconds, average ari : [", mean(na.omit(aris)), "]\n")
# total_ari = adjustedRandIndex(all_found_labels, head(t(lbls), length(all_found_labels)))
# cat("Total ari : [", total_ari, "]\n")

