####################################
### Classification of Spiral Data ##
####################################

N <- 200 # number of points per class
K <- 5 # number of classes

num.neurons <- 150
l <- 2

scale <- 1
prop <- 0.75
plot <- TRUE
NoiseX <- FALSE

# Loading functions
source("FeedForwardNN.R")
source("BackPropN.R")
source("TrainNN.R")
source("Group.R")
source("mom.R")
source("huber_loss.R")
source("huber_derivative.R")
source("l1_derivative.R")
##################################
X <- data.frame() # data matrix (each row = single example)
C <- data.frame() # class labels

for (j in (1:K)){
  r <- seq(0.05,1,length.out = N) # radius
  t <- seq((j-1)*3.7,(j)*3.7, length.out = N) + rnorm(N, sd = 0.25) # theta
  Xtemp <- data.frame(x1 =r*sin(t) , x2 = r*cos(t)) 
  ytemp <- data.frame(matrix(j, N, 1))
  X <- rbind(X, Xtemp)
  C <- rbind(C, ytemp)
}

data <- cbind(X,C)
colnames(data) <- c(colnames(X), 'label')

x_min <- min(X[,1])-0.2; x_max <- max(X[,1])+0.2
y_min <- min(X[,2])-0.2; y_max <- max(X[,2])+0.2

# lets visualize the data:
if(isTRUE(plot)){
  library(ggplot2)
  ggplot(data) + geom_point(aes(x=x1, y=x2, color = as.character(label)), size = 2) + theme_bw(base_size = 15) +
    xlim(x_min, x_max) + ylim(y_min, y_max) +
    ggtitle('Spiral data with five classes') +
    coord_fixed(ratio = 0.8) +
    theme(axis.ticks=element_blank(), panel.grid.major = element_blank(), panel.grid.minor = element_blank(), 
          legend.position = 'none')+xlab("Coordinate 1") + ylab("Coordinate 2")
}

X <- as.matrix(X)/max(abs(X))
########################################
Y <- matrix(0, N*K, K)

for (i in 1:(N*K)){
  Y[i, C[i,]] <- 1
}
set.seed(20210314)
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
  X[noise, ] <- X[noise, ] + matrix(rnorm(dim(X)[2]*length(noise), 0, 1), nrow=length(noise), ncol=dim(X)[2])
}

num.obs <- dim(X)[1]
num.par <- dim(X)[2]

b <- floor(dim(X)[1]*0.15)

E <- 20000
i.num <- floor(num.obs/b)*E

P <- rep(num.neurons,l)
P <- c(num.par,P,K)
################################################################################
set.seed(NULL)
# Softmax  
alpha <- 1e-2
repeat{
  Pre_Para.softmax <- TrainNN(y=Y,X,P=P,alpha=alpha,iteration=i.num,random=TRUE,batch=b,MOM=FALSE,k=3,loss.f="ls",q=NULL,bias=TRUE,class=TRUE,beta=scale)
  train.test <- FeedForwardNN(X,para=Pre_Para.softmax[[1]],class=TRUE,class.score=TRUE)
  
  if(any(is.na(train.test))|isTRUE(train.test=="NaN")|isTRUE(any(is.na(train.test)))|isTRUE(is.na(train.test))|isTRUE(train.test==Inf)|isTRUE(train.test==-Inf)|isTRUE(identical(train.test,integer(0)))){
    alpha <- alpha/2
    next
  }else{
    break
  }
}

Prediction.class <- FeedForwardNN(X.test,para=Pre_Para.softmax[[1]],class=TRUE,class.score=FALSE)
alpha.softmax <- alpha
LossTracking.ls <- Pre_Para.softmax[[2]]
accuracy.softmax <- mean(Prediction.class==(CT))

# MOM 
Blocks <- c(3,5,7,9,11,13,15,17)
Prediction.mom <- c()
alpha.mom <- c() 
LossTracking.mom <- list()
for(i in 1:length(Blocks)){
  alpha <- 1e-2
  repeat{
    Pre_Para.mom <- TrainNN(y=Y,X=X,P=P,alpha=alpha,iteration=i.num,random=TRUE,batch=b,MOM=TRUE,k=Blocks[i],loss.f="ls",q=NULL,bias=TRUE,class=TRUE,beta=scale)
    train.test <- FeedForwardNN(X,para=Pre_Para.mom[[1]],class=TRUE,class.score=TRUE)
    
    if(any(is.na(train.test))|isTRUE(train.test=="NaN")|isTRUE(any(is.na(train.test)))|isTRUE(is.na(train.test))|isTRUE(train.test==Inf)|isTRUE(train.test==-Inf)|isTRUE(identical(train.test,integer(0)))){
      alpha <- alpha/2
      next
    }else{
      break
    }
  }
  
  Prediction.class.mom <- FeedForwardNN(X.test,para=Pre_Para.mom[[1]],class=TRUE,class.score=FALSE)
  LossTracking.mom[[i]] <- Pre_Para.mom[[2]]
  accuracy.mom <- mean(Prediction.class.mom==(CT))
  Prediction.mom <- c(Prediction.mom,accuracy.mom)
  alpha.mom <- c(alpha.mom,alpha)
}

accuracy.softmax
max(Prediction.mom)

