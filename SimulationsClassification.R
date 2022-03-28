####################################
### Classification of Spiral Data ##
####################################

# Settings

## A positive integer indicates the numebr of classes.
K <- 5 

## A positive integer indicates the number of samples for each class.
N <- 200 

## A positive integer indicates the number of neurons in each hidden layer of 
## the neural network.
num.neurons <- 150 

## A positive integer indicates the number of layers for the neural network.
l <- 2 

## A positive value within [0,1] indicates the propostion of informative 
## samples.
prop <- 1 

## A Boolean value indicates whether to corrupt the input vectors.
NoiseX <- FALSE 

## A positive value within [0,1] indicates the batch size for stochastic 
## gradient descent algorithm.
BatchSize <- 1

## A positive integer indicates the number of epoch to update the gradient of 
## the neural network.
E <- 2000

## A positive value indicates the learning rate for the stochastic gradient 
## descent algorithm with softmax loss. 
alphaSoftmax <- 1e-1

## A positive value indicates the learning rate for the stochastic gradient 
## descent algorithm with DeepMoM structure. 
alphaMoM <- 1e-1

## A vector of positive integers indicates the number of blocks for DeepMoM.
Blocks <- c(3, 5, 7, 9, 11)

## A positive value to scale the initial values for updating the gradients of 
## the neural network.
scale <- 1 

## A positive value to set the random seed for shuffling the data for 
## reproducible research. 
seed <- 202101

# Loading required functions

source("./AdditionalFunctions/FeedForwardNN.R")
source("./AdditionalFunctions/BackPropNN.R")
source("./AdditionalFunctions/TrainNN.R")
source("./AdditionalFunctions/GroupK.R")
source("./AdditionalFunctions/Mom.R")
source("./AdditionalFunctions/HuberLoss.R")
source("./AdditionalFunctions/HuberDerivative.R")
source("./AdditionalFunctions/L1Derivative.R")

# Generating spiral data

X <- data.frame() 
C <- data.frame() 

for (j in (1:K)){
  r <- seq(0.05,1,length.out = N) 
  t <- seq((j-1)*3.7, (j)*3.7, length.out=N) + rnorm(N, sd=0.25) 
  Xtemp <- data.frame(x1 =r*sin(t) , x2 = r*cos(t)) 
  ytemp <- data.frame(matrix(j, N, 1))
  X <- rbind(X, Xtemp)
  C <- rbind(C, ytemp)
}

data <- cbind(X,C)
colnames(data) <- c(colnames(X), 'label')

x_min <- min(X[,1])-0.2; x_max <- max(X[,1])+0.2
y_min <- min(X[,2])-0.2; y_max <- max(X[,2])+0.2

X <- as.matrix(X)/max(abs(X))

# Formalize outputs

Y <- matrix(0, N*K, K)

for (i in 1:(N*K)){
  Y[i, C[i,]] <- 1
}

set.seed(seed)
train.index <- sample(c(1:(N*K)), floor(0.5*N*K),replace=FALSE)

X.test <- X[-train.index,]
CT <- C[-train.index,]

X <- X[train.index,]
Y <- Y[train.index,]

good <- sample(c(1:dim(X)[1]), floor(dim(X)[1]*prop), replace=FALSE)

C.test <- C[train.index,]

total.index <- c(1:dim(X)[1])

if(isTRUE(NoiseX==FALSE)){
  bad <- total.index[-good]
  for(i in bad){
    s <- c(1:K)
    C.test[i] <- sample(s[-C.test[i]], 1, replace=FALSE)
  }
}

Y <- matrix(0, dim(X)[1], K)

for (i in 1:(dim(X)[1])){
  Y[i, C.test[i]] <- 1
}

if (isTRUE(NoiseX)){
  noise <- sample(c(1:dim(X)[1]), floor((1-prop)*dim(X)[1]), replace = FALSE)
  X[noise, ] <- matrix(rnorm(dim(X)[2]*length(noise), 5, 1), 
                                    nrow=length(noise), ncol=dim(X)[2])
}

X[noise, ] <- as.matrix(X[noise, ])/max(abs(X[noise, ]))

num.obs <- dim(X)[1]
num.par <- dim(X)[2]

b <- floor(dim(X)[1]*BatchSize)

i.num <- floor(num.obs/b)*E

P <- rep(num.neurons,l)
P <- c(num.par,P,K)
set.seed(NULL)

# Softmax  

alpha <- alphaSoftmax
repeat{
  Pre_Para.softmax <- TrainNN(y=Y,X,P=P,alpha=alpha,iteration=i.num,random=FALSE,
                              batch=b,MOM=FALSE,k=3,loss.f="ls",q=NULL,
                              bias=TRUE,class=TRUE,beta=scale, para=NULL)
  train.test <- FeedForwardNN(X, para=Pre_Para.softmax[[1]], class=TRUE,
                              class.score=TRUE)
  
  if(any(is.na(train.test))|isTRUE(train.test=="NaN")|
     isTRUE(any(is.na(train.test)))|isTRUE(is.na(train.test))|
     isTRUE(train.test==Inf)|isTRUE(train.test==-Inf)|
     isTRUE(identical(train.test,integer(0)))){
    alpha <- alpha/2
    next
  }else{
    break
  }
}

Prediction.class <- FeedForwardNN(X.test,para=Pre_Para.softmax[[1]],class=TRUE,
                                  class.score=FALSE)
alpha.softmax <- alpha
LossTracking.ls <- Pre_Para.softmax[[2]]
accuracy.softmax <- mean(Prediction.class==(CT))

# MOM 

Prediction.mom <- c()
alpha.mom <- c() 
LossTracking.mom <- list()
for(i in 1:length(Blocks)){
  alpha <- alphaMoM
  repeat{
    Pre_Para.mom <- TrainNN(y=Y,X=X,P=P,alpha=alpha,iteration=i.num,random=FALSE,
                            batch=b,MOM=TRUE,k=Blocks[i],loss.f="ls",q=NULL,
                            bias=TRUE,class=TRUE,beta=scale, para=NULL)
    train.test <- FeedForwardNN(X,para=Pre_Para.mom[[1]],class=TRUE,
                                class.score=TRUE)
    
    if(any(is.na(train.test))|isTRUE(train.test=="NaN")|
       isTRUE(any(is.na(train.test)))|isTRUE(is.na(train.test))|
       isTRUE(train.test==Inf)|isTRUE(train.test==-Inf)|
       isTRUE(identical(train.test,integer(0)))){
      alpha <- alpha/2
      next
    }else{
      break
    }
  }
  
  Prediction.class.mom <- FeedForwardNN(X.test,para=Pre_Para.mom[[1]],
                                        class=TRUE,class.score=FALSE)
  LossTracking.mom[[i]] <- Pre_Para.mom[[2]]
  accuracy.mom <- mean(Prediction.class.mom==(CT))
  Prediction.mom <- c(Prediction.mom,accuracy.mom)
  alpha.mom <- c(alpha.mom,alpha)
}

accuracy.softmax
max(Prediction.mom)

#save.image("100.RData")




