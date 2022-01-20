num.sim <- 10

# Loading functions
source("~/Desktop/TCGA/Code/FeedForwardNN.R")
source("~/Desktop/TCGA/Code/BackPropN.R")
source("~/Desktop/TCGA/Code/TrainNN.R")
source("~/Desktop/TCGA/Code/Group.R")
source("~/Desktop/TCGA/Code/mom.R")
source("~/Desktop/TCGA/Code/huber_loss.R")
source("~/Desktop/TCGA/Code/huber_derivative.R")
source("~/Desktop/TCGA/Code/l1_derivative.R")

A.soft <- rep(0,num.sim)
A.mom <- rep(0,num.sim)

#data <- as.matrix(read.csv("DataMiRna.csv", sep=","))
load("~/Desktop/TCGA/data/TCGAlist.RData")

X_ov <- as.matrix(TCGA.data$OV$RC)
y_ov <- rep(1, dim(X_ov)[1])
data_ov <- cbind(y_ov, X_ov)

X_sarc <- as.matrix(TCGA.data$SARC$RC)
y_sarc <- rep(2, dim(X_sarc)[1])
data_sarc <- cbind(y_sarc, X_sarc)

X_kirc <- as.matrix(TCGA.data$KIRC$RC)
y_kirc <- rep(3, dim(X_kirc)[1])
data_kirc <- cbind(y_kirc, X_kirc)

X_luad <- as.matrix(TCGA.data$LUAD$RC)
y_luad <- rep(4, dim(X_luad)[1])
data_luad <- cbind(y_luad, X_luad)

X_skcm <- as.matrix(TCGA.data$SKCM$RC)
y_skcm <- rep(5, dim(X_skcm)[1])
data_skcm <- cbind(y_skcm, X_skcm)

#X_acc <- as.matrix(TCGA.data$ACC$RC)
#y_acc <- rep(6, dim(X_acc)[1])
#data_acc <- cbind(y_acc, X_acc)


X_esca <- as.matrix(TCGA.data$ESCA$RC)
y_esca <- rep(6, dim(X_esca)[1])
data_esca <- cbind(y_esca, X_esca)


#X_ucs <- as.matrix(TCGA.data$UCS$RC)
#y_ucs <- rep(8, dim(X_ucs)[1])
#data_ucs <- cbind(y_ucs, X_ucs)



X_laml <- as.matrix(TCGA.data$LAML$RC)
y_laml <- rep(7, dim(X_laml)[1])
data_laml <- cbind(y_laml, X_laml)


#X_chol <- as.matrix(TCGA.data$CHOL$RC)
#y_chol <- rep(10, dim(X_chol)[1])
#data_chol <- cbind(y_chol, X_chol)



data <- rbind(data_ov,
              data_sarc,
              data_kirc,
              data_luad,
              data_skcm,
              #data_acc,
              data_esca,
              #data_ucs,
              #data_chol,
              data_laml)




set.seed(202101)
data <- data[sample(c(1:dim(data)[1]),dim(data)[1], replace=FALSE),]
set.seed(NULL)

set.seed(202102)
GG <- group(num.sim, dim(data)[1])
set.seed(NULL)

num.neurons <- 150
l <- 2
check <- FALSE
E <- 20000

c.loss.soft <- 1e-100
c.update.soft <- 1e-100

c.loss.mom <- 1e-100
c.update.mom <- 1e-100

scale <- 1

for (sim in 1:num.sim){
  X <- as.matrix(data[,-1])
  
  for (i in 1:dim(X)[2]){
    s <- sum(abs(X[,i]))
    if(isTRUE(s!=0)){
      X[,i] <- as.vector(X[,i])/s*1e+6
    }
  }
  
  X <- X/max(abs(X))
  y <- as.vector(data[,1])
  
  num.obs <- dim(X)[1]
  
  IndexT <- GG[[sim]]
  
  X.original <- X[-IndexT,]
  y.original <- y[-IndexT]
  
  X.test <- X[IndexT,]
  y.test <- y[IndexT]
  
  ################################################################################
  C <- y.original
  CT <- y.test
  
  X <- X.original
  
  K <- length(unique(y))
  
  Y <- matrix(rep(0, dim(X)[1]*K), ncol=K)
  
  for (i in 1:(dim(X)[1])){
    Y[i, C[i]] <- 1
  }
  
  num.obs <- dim(X)[1]
  num.par <- dim(X)[2]
  
  b <- floor(num.obs*0.15)
  
  num.i <- floor(num.obs/b)*E
  
  P <- rep(num.neurons,l)
  P <- c(num.par,P,K)
  
  par(mfrow=c(1,1))
  ################################################################################
  set.seed(NULL)
  # Softmax
  alpha <- 1e-2
  repeat{
    Pre_Para.softmax <- TrainNN(y=Y,X,P=P,alpha=alpha,iteration=num.i,random=TRUE,batch=b,stop.i.loss=c.loss.soft,stop.i.update=c.update.soft,MOM=FALSE,k=3,loss.f="ls",q=NULL,bias=TRUE,class=TRUE,beta=scale,qs=NULL)
    train.test <- FeedForwardNN(X,para=Pre_Para.softmax[[1]],class=TRUE,class.score=TRUE)
    
    if(any(is.na(train.test))|isTRUE(train.test=="NaN")|isTRUE(is.na(train.test))|isTRUE(train.test==Inf)|isTRUE(train.test==-Inf)|isTRUE(identical(train.test,integer(0)))){
      alpha <- alpha/2
      next
    }else{
      break
    }
  }
  
  Prediction.class <- FeedForwardNN(X.test,para=Pre_Para.softmax[[1]],class=TRUE,class.score=FALSE)
  #Prediction.class <- Prediction.class-1
  alpha.softmax <- alpha
  LossTracking.ls <- Pre_Para.softmax[[2]]
  accuracy.softmax <- mean(Prediction.class==(CT))
  
  save.image(paste0(sim, "ls.RData"))
  
  # MOM 
  b <- floor(num.obs*0.15)
  num.i <- floor(num.obs/b)*E
  
  Blocks <- c(3,5,7,9,11)
  Prediction.mom <- c()
  alpha.mom <- c() 
  LossTracking.mom <- list()
  for(i in 1:length(Blocks)){
    alpha <- 1e-2
    repeat{
      Pre_Para.mom <- TrainNN(y=Y,X=X,P=P,alpha=alpha,iteration=num.i,random=TRUE,batch=b,stop.i.loss=c.loss.mom,stop.i.update=c.update.mom,MOM=TRUE,k=Blocks[i],loss.f="ls",q=NULL,bias=TRUE,class=TRUE,beta=0.1*scale,qs=NULL)
      train.test <- FeedForwardNN(X,para=Pre_Para.mom[[1]],class=TRUE,class.score=TRUE)
      
      if(any(is.na(train.test))|isTRUE(train.test=="NaN")|isTRUE(is.na(train.test))|isTRUE(train.test==Inf)|isTRUE(train.test==-Inf)|isTRUE(identical(train.test,integer(0)))){
        alpha <- alpha/2
        next
      }else{
        break
      }
    }
    
    Prediction.class.mom <- FeedForwardNN(X.test,para=Pre_Para.mom[[1]],class=TRUE,class.score=FALSE)
    #Prediction.class.mom <- Prediction.class.mom-1
    LossTracking.mom[[i]] <- Pre_Para.mom[[2]]
    accuracy.mom <- mean(Prediction.class.mom==(CT))
    Prediction.mom <- c(Prediction.mom,accuracy.mom)
    alpha.mom <- c(alpha.mom,alpha)
    save.image(paste0(Blocks[i],sim, "mom.RData"))
  }
  
  #accuracy.softmax
  #max(Prediction.mom)
  
  A.soft[sim] <- accuracy.softmax
  A.mom[sim] <- max(Prediction.mom)
}

mean(A.soft)
mean(A.mom)


#save.image("all.RData")

class.mom <- FeedForwardNN(X,para=Pre_Para.mom[[1]],class=TRUE,class.score=FALSE)
mean(class.mom==(C))