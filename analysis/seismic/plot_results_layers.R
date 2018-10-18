require(reshape2)
require(ggplot2)
require(RColorBrewer)
library("RColorBrewer") 
#input_path= "~/Desktop/1savigp/results-amir/Archive/var_div1/"
input_path = "seismic_24-May-2018_16h32m05s_17885/" # Seemed to have had some numerical issues 
#input_path = "seismic_29-May-2018_11h39m43s_2681/" # initialized posterior variances to prior variances 
#input_path = "seismic_29-May-2018_15h53m28s_6630/" # gradients wrt mean and covars corrected 
#input_path = "/Users/ebonilla/Documents/research/savigp/code/results/seismic_30-May-2018_17h00m33s_7555/"
#input_path = "/Users/ebonilla/Documents/research/savigp/code/results/seismic_31-May-2018_11h25m37s_19628/"
#input_path = "seismic_01-Jun-2018_13h33m59s_3234/" # Optimizing hyper
#input_path = "seismic_04-Jun-2018_12h16m39s_5787/" # Joint optimization mog+hyper
 

train_fname = paste0(input_path, "train.csv")
pred_fname = paste0(input_path, "predictions.csv")
mcmc_fname = "mcmc_results.csv"

# read results files
pred = read.csv(pred_fname)
x =   read.csv(train_fname)$X_0
pred = cbind(pred, x)
mcmc_pred = read.csv(mcmc_fname)
mcmc_pred = cbind(mcmc_pred, x)

# for depth
depth = pred[,c("predicted_Y_0", "predicted_Y_1", "predicted_Y_2", "predicted_Y_3", "x")]
depth_var = pred[,c("predicted_variance_0", "predicted_variance_1", "predicted_variance_2", "predicted_variance_3", "x")]
depth = melt(depth, id = "x")
depth_var = melt(depth_var, id = "x", value.name = "var")
d = cbind(depth, depth_var)
d$value = -d$value
d_high = d$value + sqrt(d$var)
d_low = d$value - sqrt(d$var)


# mcmc baseline for depth
mcmc_depth = mcmc_pred[, c("mean_depth_0", "mean_depth_1", "mean_depth_2", "mean_depth_3", "x")]
mcmc_depth_std = mcmc_pred[, c("std_depth_0", "std_depth_1", "std_depth_2", "std_depth_3", "x")]
mcmc_depth = melt(mcmc_depth, id = "x")
mcmc_depth_std = melt(mcmc_depth_std, id = "x", value.name = "std")
mcmc_d = cbind(mcmc_depth, mcmc_depth_std)
mcmc_d$value = -mcmc_d$value
mcmc_d_high = mcmc_d$value + mcmc_d$std
mcmc_d_low = mcmc_d$value - mcmc_d$std


col = brewer.pal(4, "OrRd")[4]
ggplot(d, aes(x = x, y = value, group=variable, color=col)) + 
  geom_line()+
  geom_ribbon(aes(ymin=d_low,ymax=d_high),alpha=0.3)  +
  theme_bw() +
  xlab("Sensor location (m)") +
  ylab("Height (m)") +
  theme(legend.direction = "horizontal", legend.position = "none", legend.box = "horizontal", 
        axis.line = element_line(colour = "black"),
        panel.grid.major=element_blank(), 
        panel.grid.minor=element_blank(),      
        panel.border = element_blank(),
        text=element_text(size=10),
        legend.title=element_blank(),
        legend.margin=margin(t = -0.2, unit='cm') ,
        plot.margin = unit(x = c(0.01, 0.01, 0.01, 0.01), units = "cm"),        
        #      axis.text.x = element_blank(),
        legend.key = element_blank(),
        panel.background = element_rect(fill=NA, color ="black")
  ) +
  guides(fill = guide_legend(keywidth = 0.5, keyheight = 0.5, nrow = 2)) +
  geom_line(data = mcmc_d, aes(x = x, y = value, group = variable, color ="black"))
  
ggsave("depth.pdf", height = 8, width = 8, units = "cm")

#### for velocity
vel = pred[,c("predicted_Y_4", "predicted_Y_5", "predicted_Y_6", "predicted_Y_7", "x")]
vel_var = pred[,c("predicted_variance_4", "predicted_variance_5", "predicted_variance_6", "predicted_variance_7", "x")]
require(reshape2)
vel = melt(vel, id = "x")
vel_var = melt(vel_var, id = "x", value.name = "var")
v = cbind(vel, vel_var)
v$value = v$value
v_high = v$value + sqrt(v$var)
v_low = v$value - sqrt(v$var)

# MCMC Baseline for velocity
mcmc_vel = mcmc_pred[, c("mean_vel_0", "mean_vel_1", "mean_vel_2", "mean_vel_3", "x")]
mcmc_vel_std = mcmc_pred[, c("std_vel_0", "std_vel_1", "std_vel_2", "std_vel_3", "x")]
mcmc_vel = melt(mcmc_vel, id = "x")
mcmc_vel_std = melt(mcmc_vel_std, id = "x", value.name = "std")
mcmc_v = cbind(mcmc_vel, mcmc_vel_std)
mcmc_v$value = mcmc_v$value
mcmc_v_high = mcmc_v$value + mcmc_v$std
mcmc_v_low = mcmc_v$value - mcmc_v$std

col = brewer.pal(4, "OrRd")[4]
require(ggplot2)
ggplot(v, aes(x = x, y = value, group=variable, color=col)) + 
  geom_line()+
  geom_ribbon(aes(ymin=v_low,ymax=v_high),alpha=0.3)  +
  theme_bw() +
  xlab("Sensor location (m)") +
  ylab("Velocity (m/s)") +
  theme(legend.direction = "horizontal", legend.position = "none", legend.box = "horizontal", 
        axis.line = element_line(colour = "black"),
        panel.grid.major=element_blank(), 
        panel.grid.minor=element_blank(),      
        panel.border = element_blank(),
        text=element_text(size=10),
        legend.title=element_blank(),
        legend.margin=margin(t = -0.2, unit='cm') ,
        plot.margin = unit(x = c(0.01, 0.01, 0.01, 0.01), units = "cm"),        
        #      axis.text.x = element_blank(),
        legend.key = element_blank(),
        panel.background = element_rect(fill=NA, color ="black")
  ) +
  guides(fill = guide_legend(keywidth = 0.5, keyheight = 0.5, nrow = 2)) +
  geom_line(data = mcmc_v, aes(x = x, y = value, group = variable, color ="black"))



ggsave("vel.pdf", height = 8, width = 8, units = "cm")
 