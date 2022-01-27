#Script to calculate F1 and F2 class overlap scores of activity and resource features
library("ECoL")
library(logr)

filename = paste(gsub(":", "-", Sys.time()),"_file.log",sep="")
log_open(filename)
log_print("---- Running check_overlap.r ----")

root_dir <- "../event_logs/"
event_logs <- list.files(root_dir, recursive=FALSE) 
na <-c("NA","NA")
names(na) <- c("mean","sd")

#ACTIVITY
activity_f1_mean = c()
activity_f1_std = c()
activity_f2_mean = c()
activity_f2_std = c()

#RESOURCE
resource_f1_mean = c()
resource_f1_std = c()
resource_f2_mean = c()
resource_f2_std = c()

for(log in event_logs){
	print(paste("---- Checking Overlap Score in", log, " ----"))
	dataset = paste(root_dir,log,sep='')
	data <- read.csv(dataset)

	#ACTIVITY
	tryCatch(
	    expr = {
			y <- featurebased(activity ~ ., data)
			activity_f1_mean<-c(activity_f1_mean,y$F1[1])
			activity_f1_std<-c(activity_f1_std,y$F1[2])
			activity_f2_mean<-c(activity_f2_mean,y$F2[1])
			activity_f2_std<-c(activity_f2_std,y$F2[2])
	    },
	    error = function(e){ 
	    	activity_f1_mean<<-c(activity_f1_mean,na[1])
			activity_f1_std<<-c(activity_f1_std,na[2])
			activity_f2_mean<<-c(activity_f2_mean,na[1])
			activity_f2_std<<-c(activity_f2_std,na[2])
	    	log_print(paste("[ERROR][",log,"][ACTIVITY]: ", e))
	    }
	)

	#RESOURCE
	tryCatch(
	    expr = {
			z <- featurebased(resource ~ ., data)
			resource_f1_mean<-c(resource_f1_mean,z$F1[1])
			resource_f1_std<-c(resource_f1_std,z$F1[2])
			resource_f2_mean<-c(resource_f2_mean,z$F2[1])
			resource_f2_std<-c(resource_f2_std,z$F2[2])
	    },
	    error = function(e){ 
			resource_f1_mean<<-c(resource_f1_mean,na[1])
			resource_f1_std<<-c(resource_f1_std,na[2])
			resource_f2_mean<<-c(resource_f2_mean,na[1])
			resource_f2_std<<-c(resource_f2_std,na[2])
	    	log_print(paste("[ERROR][",log,"][RESOURCE]: ", e))
	    }
	)
}

df <- data.frame(event_logs, activity_f1_mean, activity_f1_std, activity_f2_mean, activity_f2_std, resource_f1_mean, resource_f1_std, resource_f2_mean, resource_f2_std)
write.csv(df,"class_overlap_scores.csv")
print('CSV file written successfully')
log_close()