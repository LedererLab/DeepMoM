#############
### Group ###
#############

group <- function(K, n){
  # Description :
  #               Partition the data into some groups.
  # Usage : 
  #         group(K, n)
  # Arguments : 
  #   K : A positive integer indicates the number of partitions of the data.
  #   n : A positive integer indicates the total number of data.
  # Returns : 
  #   A list contains the partition of data.
  
  output <- list()
  total <- c(1:n)
  if (K > 1){
    for (i in 1:(K-1)){
      output[[i]] <- sample(total,floor(n/K),replace=FALSE)
      total <- total[!total%in%output[[i]]]
    }
    output[[K]] <- total
  } else {
    output[[1]] <- total
  }
  
  return(output)
}

