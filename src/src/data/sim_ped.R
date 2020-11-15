require(optparse)
library(dplyr)
library(abind)



option_list = list(
    make_option(c(''))
)


n <- dim()
x1 <- runif(n, -2, 3)
x2 <- runif(n, 0, 5)

sim_dataset <- data.frame(item=label, x1=x1, x2=x2)

f0 <- function(t){
    -0.1 * (t - 4) ^ 2
}


form <- ~ -0.5 + f0(t) + 0.25 * x1 - 0.3 * x2 + 0.5 * (item == 1) + -1 * (item == 2)
cens_times <- rexp(n, 0.02)
survival_data <- sim_pexp(form, sim_dataset, seq(0, 10, by=.2)) %>%
    mutate(status = ifelse(time > cens_times, 0, status)) %>%
    mutate(time = ifelse(time > cens_times, cens_times, time)) %>%
    mutate(time = round(time, 2)) %>%
    mutate(time = ifelse(time == 0L, time + 0.01, time))

hist(survival_data$time)
mean(survival_data$status)
