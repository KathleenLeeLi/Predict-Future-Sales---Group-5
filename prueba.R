library(feather) # data import
library(data.table) # data handle
library(rpart) # decision tree method
library(rpart.plot) # tree plot
library(party) # decision tree method
library(forecast) # forecasting methods
library(ggplot2) # visualizations
library(ggforce) # visualization tools
library(plotly) # interactive visualizations
library(grid) # visualizations
library(animation) # gif

#install.packages("feather") # data import
install.packages("data.table") # data handle
install.packages("rpart") # decision tree method
install.packages("rpart.plot") # tree plot
install.packages("party") # decision tree method
install.packages("forecast") # forecasting methods
install.packages("ggplot2") # visualizations
install.packages("ggforce") # visualization tools
install.packages("plotly") # interactive visualizations
install.packages("grid") # visualizations
install.packages("animation") # gif

install.packages()



setwd("D:/Mis Documentos/Data Science Certificate/Assignments/Group Assignment")

prueba<- read.csv("sales_train_prueba2.csv")
View(prueba)
attach(prueba)
names(prueba)


dim(n_date2)
n_date2 <- unique(prueba['date2'])
length(n_date2)
period <- 48

View(data_msts)
length(data_msts)

data_msts <- msts(item_cnt_day_total, seasonal.periods = c(period, period*7))

data_ts <- ts(item_cnt_day_total, freq = period * 7)
decomp_ts <- stl(data_ts, s.window = "periodic", robust = TRUE)$time.series

N <- nrow(prueba)
window <- (N / period) - 1 # number of days in train set minus lag

new_load <- rowSums(decomp_ts[, c(1,3)]) # detrended load
lag_seas <- decomp_ts[1:(period*window), 1] #
length(lag_seas)


matrix_train2 <- data.table(Load = tail(new_load, window*period),
                           data_msts,
                           Lag = lag_seas)

matrix_train3 <- data.table(Load = tail(new_load, window*period),
                            data_msts,
                            Lag = lag_seas,
                            shop_id)

View(matrix_train3)

tree_2 <- rpart(Load ~ ., data = matrix_train2)

paste("Number of splits: ", tree_2$cptable[dim(tree_2$cptable)[1], "nsplit"])

datas2 <- data.table(Load = c(matrix_train2$Load,
                             predict(tree_2)),
                    Time = rep(1:length(matrix_train2$Load), 2),
                    Type = rep(c("Real", "RPART"), each = length(matrix_train2$Load)))

ggplot(datas2, aes(Time, Load, color = Type)) +
  geom_line(size = 0.8, alpha = 0.75) +
  labs(y = "Detrended load", title = "Fitted values from RPART tree")

mape(matrix_train2$Load, predict(tree_2))

tree_1 <- rpart(Load ~ ., data = matrix_train3)

paste("Number of splits: ", tree_1$cptable[dim(tree_1$cptable)[1], "nsplit"])

datas <- data.table(Load = c(matrix_train3$Load,
                             predict(tree_1)),
                    Time = rep(1:length(matrix_train3$Load), 2),
                    Type = rep(c("Real", "RPART"), each = length(matrix_train3$Load)))

ggplot(datas, aes(Time, Load, color = Type)) +
  geom_line(size = 0.8, alpha = 0.75) +
  labs(y = "Detrended load", title = "Fitted values from RPART tree")

mape <- function(real, pred){
  return(100 * mean(abs((real - pred)/real))) # MAPE - Mean Absolute Percentage Error
}


mape(matrix_train3$Load, predict(tree_1))

#------------
tree3 <- rpart(Load ~ ., data = matrix_train3,
                control = rpart.control(minsplit = 2,
                                        maxdepth = 30,
                                        cp = 0.000001))
tree3$cptable[dim(tree3$cptable)[1], "nsplit"] # Number of splits

datas3 <- data.table(Load = c(matrix_train3$Load,
                             predict(tree3)),
                    Time = rep(1:length(matrix_train3$Load), 2),
                    Type = rep(c("Real", "RPART"), each = length(matrix_train3$Load)))

ggplot(datas3, aes(Time, Load, color = Type)) +
  geom_line(size = 0.8, alpha = 0.75) +
  labs(y = "Detrended load", title = "Fitted values from RPART")

mape(matrix_train3$Load, predict(tree3))


#-------- Test

test_data<- read.csv("test.csv")

View(test_data)

unique(test_data$shop_id)

test_lag <- decomp_ts[((period*window)+1):N, 1]
data_msts

matrix_test <- data.table(data_msts,
                          Lag = test_lag,
                          shop_id=unique(test_data$shop_id))

pred<- predict(tree3, matrix_test)

data_for <- data.table(Load = c(prueba$item_cnt_day_total, data_test$value, for_rpart),
                       Date = c(data_train$date_time, rep(data_test$date_time, 2)),
                       Type = c(rep("Train data", nrow(data_train)),
                                rep("Test data", nrow(data_test)),
                                rep("Forecast", nrow(data_test))))

####################################################################

DT <- as.data.table(read_feather("DT_load_17weeks"))
View(DT)

length(n_date)
n_date <- unique(DT[, date])
period <- 48

data_train <- DT[date %in% n_date[43:63]]
data_test <- DT[date %in% n_date[64]]

View(data_train)

data_msts <- msts(data_train$value, seasonal.periods = c(period, period*7))

View(data_msts)

data_ts <- ts(data_train$value, freq = period * 7)
decomp_ts <- stl(data_ts, s.window = "periodic", robust = TRUE)$time.series


N <- nrow(data_train)
window <- (N / period) - 1 # number of days in train set minus lag

new_load <- rowSums(decomp_ts[, c(1,3)]) # detrended load
lag_seas <- decomp_ts[1:(period*window), 1] # seasonal part of time series as lag feature

K <- 2
fuur <- fourier(data_msts, K = c(K, K))

matrix_train <- data.table(Load = tail(new_load, window*period),
                           fuur[(period + 1):N,],
                           Lag = lag_seas)

