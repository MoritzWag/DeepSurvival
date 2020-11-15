library(pammtools)
library(dplyr)
library(abind)
use_implementation("tensorflow")
np <- import("numpy")
seed <- 277

train_pcs <- np$load("mnist_toy/input/train_alternative.npy") 
train_labels <- np$load("mnist_toy/input/train_alternative_labels.npy")
test_pcs <- np$load("mnist_toy/input/test_alternative.npy") 
test_labels <- np$load("mnist_toy/input/test_alternative_labels.npy")

pc_train <- train_pcs[train_labels %in% c(0, 1, 2, 10), 
                      1: dim(train_pcs)[2], 1:dim(train_pcs)[3]]
pc_test <- test_pcs[test_labels %in% c(0, 1, 2), 
                    1: dim(test_pcs)[2], 1:dim(test_pcs)[3]]
pc <- abind(pc_train, pc_test, along = 1)

ltrain <- train_labels[train_labels %in% c(0, 1, 2, 10)]
ltest  <- test_labels[test_labels %in% c(0, 1, 2)]
label  <- c(ltrain, ltest)

augment <- function(points, seed = 1L) {
  set.seed(seed)
  points <- points + tf$random$uniform(dim(points), -0.005, 0.005)$numpy()
  #shuffeling <- sample(1:dim(points)[1])
  #points <- points[shuffeling, 1:dim(points)[2], 1:dim(points)[3]]
  #y <- y[shuffeling]
  points
}

n <- dim(pc)[1]
x1 <- runif(n, -2, 3)
x2 <- runif(n, 0, 5)
sim_dataset <- data.frame(item = label, x1 = x1, x2 = x2)

f0 <- function(t) {
  - 0.1 * (t - 4) ^ 2
}

set.seed(seed)
form <- ~ -0.5 + f0(t) + 0.25 * x1 - 0.3 * x2 + 0.5 * (item == 1) + -1 * (item == 2)
cens_times <- rexp(n, 0.02)
survival_data <- sim_pexp(form, sim_dataset, seq(0, 10, by = .2)) %>%
  mutate(status = ifelse(time > cens_times, 0, status)) %>%
  mutate(time   = ifelse(time > cens_times, cens_times, time)) %>%
  mutate(time   = round(time, 2)) %>%
  mutate(time   = ifelse(time == 0L, time + 0.01, time))
hist(survival_data$time)
mean(survival_data$status)
#mean(survival_data$status[survival_data$time < 10])

train_ids <- sample(1:nrow(survival_data), nrow(survival_data) - 3 * 72)
test_ids <- (1:nrow(survival_data))[!(1:nrow(survival_data) %in% train_ids)][sample(1:(3 * 72), 3 * 72)]

sdf_train <- survival_data[train_ids, ]
item_train <- sdf_train$item
sdf_train_item <- sdf_train
sdf_train <- sdf_train %>% select(-item)
sdf_test <- survival_data[test_ids, ]
item_test <- sdf_test$item
sdf_test <- sdf_test %>% select(-item)

#all_cuts <- sdf_train$time 
#all_cuts <- all_cuts[-which(all_cuts == max(all_cuts))]
#all_cuts <- all_cuts[order(all_cuts)]
#cut_samples <- sample(c(TRUE, FALSE), length(all_cuts), replace = TRUE, 
#                    prob = c(0.2, 0.8))
#cuts <- c(unique(all_cuts[cut_samples]), max(sdf_train$time))

ped_train <- as_ped(sdf_train, formula = Surv(time, status) ~ ., 
                    cut = cuts)

ped_item <- as_ped(sdf_train_item, formula = Surv(time, status) ~ ., 
                   cut = cuts)

ped_test <- as_ped(sdf_test, formula = Surv(time, status) ~ ., 
                   cut = cuts)

sdf_test2 <- sdf_test %>% mutate(time = max(time))
ped_test2 <- as_ped(sdf_test2, formula = Surv(time, status) ~ ., 
                    cut = cuts)

pc_train <- augment(pc[train_ids, , ])
pc_test <- pc[test_ids, , ]

ped_pc_train <- list(
  ped = ped_train,
  pc = pc_train,
  pc_ids = sdf_train$id,
  pc_item = item_train
)

ped_pc_test <- list(
  ped = ped_test,
  ped_complete = ped_test2,
  pc = pc_test,
  pc_ids = sdf_test$id,
  pc_item = item_test,
  original = sdf_test
)

df_artificial_test <- data.frame(
  id = rep(sdf_test$id, each = length(unique(ped_test$tend))),
  tend = rep(unique(ped_test$tend), nrow(sdf_test)),
  offset = rep(0, nrow(sdf_test) * length(unique(ped_test$tend))),
  x1 = rep(0, nrow(sdf_test) * length(unique(ped_test$tend))),
  x2 = rep(0, nrow(sdf_test) * length(unique(ped_test$tend))),
  item = rep(ped_pc_test$pc_item, each = length(unique(ped_test$tend)))
)

df_artificial_test$true_hazard <-
  exp(- 0.5 + f0(df_artificial_test$tend) + 0.25 * df_artificial_test$x1 - 
        0.3 * df_artificial_test$x2 + 
        0.5 * (df_artificial_test$item == 1) - 
        1 * (df_artificial_test$item == 2))

artificial_test <- list(
  ped = df_artificial_test %>% select(-item),
  pc = ped_pc_test$pc,
  pc_ids = sdf_test$id,
  ped_item = df_artificial_test$item,
  pc_item = ped_pc_test$pc_item
)

saveRDS(ped_pc_train, 
        paste("item/", seed, "/ped_pc_train_linear.RDS", sep = ""))
saveRDS(ped_pc_test, 
        paste("item/", seed, "/ped_pc_test_linear.RDS", sep = ""))
saveRDS(artificial_test, 
        paste("item/", seed, "/ped_pc_artificialtest_linear.RDS", sep = ""))
saveRDS(ped_item, paste("item/", seed, "/ped_item_linear.RDS", sep = ""))
saveRDS(sdf_train, paste("item/", seed, "/cox/strain.RDS", sep = ""))
saveRDS(sdf_train_item, paste("item/", seed, "/cox/strain_item.RDS", sep = ""))


