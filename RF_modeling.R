

## --------------------- Clear all environment variables for windows, tables, etc.-------------------------------------
rm(list=ls())   
graphics.off()

## ------------------------------- Load the required packages----------------------------------------------------------
suppressWarnings(library(randomForest))
suppressWarnings(library(raster))
suppressWarnings(library(rgdal))
suppressWarnings(library(sp))
suppressWarnings(library(sf))
suppressWarnings(library(ggplot2))
suppressWarnings(library(lattice))
suppressWarnings(library(base))
suppressWarnings(library(ExtractTrainData))
suppressWarnings(library(maptools))
suppressWarnings(library(rgeos))
suppressWarnings(library(colorRamps))
suppressWarnings(library(caret))
suppressWarnings(library(e1071))
suppressWarnings(library(terra))
suppressWarnings(library(mice))
suppressWarnings(library(zoo))
suppressWarnings(library(ROCR))
suppressWarnings(library(pROC))
suppressWarnings(library(OptimalCutpoints))
suppressWarnings(library(Metrics))
suppressWarnings(library(mlbench))
suppressWarnings(library(Hmisc))

## ------------------------------- Modelling data preparation----------------------------------------------------------
("rgdal_show_exportToProj4_warnings"="none")
setwd("D:/ArcFace/MLH") 
# Reading multi-spectral raster image data
image_BBS <-stack("ML.tif")
# 'for' loop to create a single wave from the stack date
plotRGB(image_BBS,4,3,2, scale=1200,stretch='hist')
i <- 0        
for (i in seq(1,32,1)) {
  fname=paste("Layer_",i,".TIF",sep="")
  layers <- image_BBS[[i]]
  writeRaster(layers,fname,
              format="GTiff",datetype='FLT4S',overwrite=TRUE)
}
# Read individual layers, stack drawing
Layer_1  <-raster("Layer_1.TIF") 
Layer_2  <-raster("Layer_2.TIF")
Layer_3  <-raster("Layer_3.TIF")
Layer_4  <-raster("Layer_4.TIF")
Layer_5  <-raster("Layer_5.TIF")
Layer_6  <-raster("Layer_6.TIF")
Layer_7  <-raster("Layer_7.TIF")
Layer_8  <-raster("Layer_8.TIF")
Layer_9  <-raster("Layer_9.TIF")
Layer_10 <-raster("Layer_10.TIF")
Layer_11 <-raster("Layer_11.TIF")
Layer_12 <-raster("Layer_12.TIF")
Layer_13 <-raster("Layer_13.TIF")
Layer_14 <-raster("Layer_14.TIF")
Layer_15 <-raster("Layer_15.TIF")
Layer_16 <-raster("Layer_16.TIF")
Layer_17 <-raster("Layer_17.TIF")
Layer_18 <-raster("Layer_18.TIF")
Layer_19 <-raster("Layer_19.TIF")
Layer_20 <-raster("Layer_20.TIF")
Layer_21 <-raster("Layer_21.TIF")
Layer_22 <-raster("Layer_22.TIF")
Layer_23 <-raster("Layer_23.TIF")
Layer_24 <-raster("Layer_24.TIF")
Layer_25 <-raster("Layer_25.TIF")
Layer_26 <-raster("Layer_26.TIF")
Layer_27 <-raster("Layer_27.TIF")
Layer_28 <-raster("Layer_28.TIF")
Layer_29 <-raster("Layer_29.TIF")
Layer_30 <-raster("Layer_30.TIF")
Layer_31 <-raster("Layer_31.TIF")
Layer_32 <-raster("Layer_32.TIF")
Alldata  <-stack(Layer_1,Layer_2,Layer_3,Layer_4,Layer_5,Layer_6,Layer_7,Layer_8,Layer_9,Layer_10,Layer_11,Layer_12,Layer_13,Layer_14,Layer_15,Layer_16,Layer_17,Layer_18,Layer_19,Layer_20,Layer_21, Layer_22,Layer_23,Layer_24,Layer_25,Layer_26,Layer_27,Layer_28,Layer_29,Layer_30,Layer_31,Layer_32)
plotRGB(Alldata,4,3,2, scale=1200,stretch='lin')
# Extraction of band values for each point of sampled data
features    <- readOGR(dsn="D:/ArcFace/Point")  # Reading point shape files
plot(features,add=TRUE)                      # Adding point shape files to RGB raster images
Out.colName <-In.colName <-"F"
MyData      <- ExtractByPoint(img = Alldata , point.shp=features, In.colName, Out.colName)

##-------------------------- RFE algorithm to select predictor variables---------------------------------------------------
MyDataClass <-as.factor(MyData$F)
set.seed(123)
inTrain     <- createDataPartition(MyDataClass, p = .75, list = FALSE)[,1]
train       <- MyData[ inTrain, ]
test        <- MyData[-inTrain, ]
trainClass  <- MyDataClass[ inTrain]
testClass   <- MyDataClass[-inTrain]
ldaProfile  <- rfe(train, trainClass,
                  
                  sizes = c(1:32),
                  
                  rfeControl = rfeControl(functions = ldaFuncs,method = "cv"))
predictors(ldaProfile)
trellis.par.set(caretTheme())

##-----------------------------Model hyperparameter optimisation----------------------------------------------------
# grid search best mtry
control       <- trainControl(method="repeatedcv", number=10, repeats=3, search="grid")
set.seed(568)
tunegrid      <- expand.grid(.mtry=c(1:32))
rf_gridsearch <- train(F~., data=MyData, method="rf", metric=metric, tuneGrid=tunegrid, trControl=control, na.action=na.roughfix)
# grid search best ntree
control       <- trainControl(method="repeatedcv", number=10, repeats=3, search="grid")
tunegrid      <- expand.grid(.mtry=c(1: 32))
modellist     <- list()
for (ntree in c(500, 1000, 1500, 2000, 2500)) {
  set.seed(2)
  fit <- train(F~., data=MyData, method="rf", metric=metric, tuneGrid=tunegrid, trControl=control, ntree=ntree, na.action=na.roughfix)
  key <- toString(ntree)
  modellist[[key]] <- fit
}
# compare results
results       <- resamples(modellist)

##------------------------------Random forest model------------------------------------------------------------------
id            <- sample(2,nrow(Mydata),replace = TRUE,prob=c(0.7,0.3))
train         <- Mydata[id==1,]
test          <- Mydata[id==2,]
# Build a random forest model based on the training set
CV            <- trainControl(method = "cv",
                   number = 10,
                   savePredictions = TRUE)
rfGrid        <- expand.grid(mtry = (1:28))
RF.train      <- randomForest(as.factor(F)~ .,data=train, trControl =CV,ntree=2500, mtry=12,tuneGrid = rfGrid,auc=TRUE,  replace=TRUE,strata=c(0,1),importance=TRUE,proximity=TRUE)
# Variable significance and OOB error curves
plot(RF.train,main = "randomforest origin")
varImpPlot(RF.train)
# Plotting the ROC curve
RF.test       <- predict(RF.train,newdata=test,type = "class")  
rf.cf         <- caret::confusionMatrix(as.factor(RF.test),as.factor(test$F))
RF.test2      <- predict(RF.train,newdata=test,type="prob") 
roc.rf        <- multiclass.roc(test$F,RF.test2[,1])
plot(roc.rf$rocs[[1]], print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),grid.col=c("green", "red"), max.auc.polygon=TRUE, main="ROC Curve",auc.polygon.col="lightblue", print.thres=TRUE)

##-----------------------------Prediction and output based on the trained random forest model--------------------------
classified   <- raster::predict(Alldata,RF.train,type='prob',progress='window')
par(mfrow=c(1,1))  # Create a 1*1 drawing area
palette      <- matlab.like(20)
plot(classified,col=palette)
plotRGB(Alldata,4,3,2,stretch='lin')
writeRaster(classified,"D:/ArcFace/classified_prob_H_1.5mg_rf,TIF",
            format="GTiff",datatype='FLT4S',overwrite=TRUE)
