require(optparse)
library(dplyr)
library(abind)
library(pammtools)
library(RcppCNPy)
library(mgcv)

seed <- 1328


option_list = list(
    make_option(c('--path'), type='character', default='../../../data/simulation/temp_storage',
                help='path to where the riskgroup information is stored'),
    make_option(c('--val_size'), type='integer', default=1000,
                help='size of validation dataset')
    
)

opt_parser = OptionParser(option_list=option_list)
args = parse_args(opt_parser)

riskgroups_train <- npyLoad(file=paste0(args$path, "/riskgroups_train.npy"))
riskgroups_test <- npyLoad(file=paste0(args$path, "/riskgroups_test.npy"))

n_train <- length(riskgroups_train)
n_test <- length(riskgroups_test)

riskgroups <- c(riskgroups_train, riskgroups_test)
n <- length(riskgroups)

x1 <- runif(n, -2, 3)
x2 <- runif(n, 0, 5)

sim_dataset <- data.frame(item=riskgroups, x1=x1, x2=x2)

f0 <- function(t){
    -0.1 * (t - 4) ^ 2
}


form <- ~ -0.5 + f0(t) + 0.25 * x1 - 0.3 * x2 + 0.5 * (item == 1) + -1 * (item == 2) + -3 * (item == 3)
cens_times <- rexp(n, 0.02)
survival_data <- sim_pexp(form, sim_dataset, seq(0, 10, by=.2)) %>%
    mutate(status = ifelse(time > cens_times, 0, status)) %>%
    mutate(time = ifelse(time > cens_times, cens_times, time)) %>%
    mutate(time = round(time, 2)) %>%
    mutate(time = ifelse(time == 0L, time + 0.01, time))

sdf_train <- survival_data[1:n_train, ]
sdf_test <- survival_data[(n_train + 1): n, ]

train_indices <- seq(1, length(riskgroups_train))
val_indices <- sample(train_indices, size=args$val_size)
sdf_val = sdf_train[val_indices, ]
sdf_train = sdf_train[-val_indices, ]


all_cuts <- sdf_train$time
all_cuts <- all_cuts[-which(all_cuts == max(all_cuts))]
all_cuts <- all_cuts[order(all_cuts)]
cut_samples <- sample(c(TRUE, FALSE), length(all_cuts), replace=TRUE, 
                      prob = c(0.2, 0.8))
cuts <- c(unique(all_cuts[cut_samples]), max(sdf_train$time))

ped_train <- as_ped(sdf_train, formula = Surv(time, status) ~ ., cut = cuts)
ped_val <- as_ped(sdf_val, formula = Surv(time, status) ~ ., cut = cuts)
ped_test <- as_ped(sdf_test, formula = Surv(time, status) ~ ., cut = cuts)

# name splines 
k = 8
# estimate splines 
pam <- gam(ped_status ~ s(tend, k=8) -1, data = ped_train)

splines_train <- as.data.frame(model.matrix(pam))
colnames(splines_train) <- paste("t", 1:(k-1), sep="")
ped_train <- ped_train %>% mutate(splines = splines_train)

splines_val <- predict(pam, ped_val, type="lpmatrix")
colnames(splines_val) <- paste("t", 1:(k-1), sep="")
ped_val <- ped_val %>% mutate(splines = splines_val)

splines_test <- predict(pam, ped_test, type="lpmatrix")
colnames(splines_test) <- paste("t", 1:(k-1), sep="")
ped_test <- ped_test %>% mutate(splines = splines_test)


# save data 
write.csv(ped_train, paste0(args$path, "/ped_train.csv"))
write.csv(ped_val, paste0(args$path, "/ped_val.csv"))
write.csv(ped_test, paste0(args$path, "/ped_test.csv"))