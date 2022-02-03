#Script to calculate F1 and F2 class overlap scores of activity and resource features
library("ECoL")
library(logr)

filename = paste(gsub(":", "-", Sys.time()),"_file.log",sep="")
log_open(filename)
log_print("---- Running check_overlap_fts.r ----")

root_dir <- "features/"
event_logs <- list.files(root_dir, recursive=FALSE) 
na <-c("NA","NA")
names(na) <- c("mean","sd")

for(log in event_logs){
	print(paste("---- Checking Overlap Score in", log, " ----"))
	dataset = paste(root_dir,log,sep='')
	data <- read.csv(dataset)
	cols <- dimnames(data)
	print(cols)
}

# df <- data.frame(event_logs, activity_f1_mean, activity_f1_std, activity_f2_mean, activity_f2_std, resource_f1_mean, resource_f1_std, resource_f2_mean, resource_f2_std)
# write.csv(df,"class_overlap_scores.csv")
print('CSV file written successfully')
log_close()