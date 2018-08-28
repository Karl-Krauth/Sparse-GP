require(reshape2)
require(RColorBrewer) 
require(ggplot2)

#input_path = "results/var_div10/"
input_path = 'results/seismic_23-Aug-2018_14h38m48s_22901/'

aa = read.csv(paste0(input_path, "predictions.csv"))
x1 =   read.csv(paste0(input_path, "train.csv"))$X_0
x2 =   x1
data = cbind(aa, x1, x2)

# for depth
depth = data[,c("predicted_Y_0", "predicted_Y_1", "predicted_Y_2", "predicted_Y_3", "x1")]
depth_var = data[,c("predicted_variance_0", "predicted_variance_1", "predicted_variance_2", "predicted_variance_3", "x2")]

depth = melt(depth, id = "x1")
depth_var = melt(depth_var, id = "x2", variable.name="depth", value.name = "var")

d = cbind(depth, depth_var)
d$value = -d$value
d_high = d$value + sqrt(d$var)
d_low = d$value - sqrt(d$var)

col = brewer.pal(4, "OrRd")[4]

ggplot(d, aes(x = x1, y = value, group=variable, color=col)) + 
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
  guides(fill = guide_legend(keywidth = 0.5, keyheight = 0.5, nrow = 2))

ggsave("depth.pdf", height = 8, width = 8, units = "cm")

# for velocity
vel = data[,c("predicted_Y_4", "predicted_Y_5", "predicted_Y_6", "predicted_Y_7", "x1")]
vel_var = data[,c("predicted_variance_4", "predicted_variance_5", "predicted_variance_6", "predicted_variance_7", "x2")]

require(reshape2)

vel = melt(vel, id = "x1")
vel_var = melt(vel_var, id = "x2", variable.name="velocity", value.name = "var")

d = cbind(vel, vel_var)
d$value = d$value
d_high = d$value + sqrt(d$var)
d_low = d$value - sqrt(d$var)

col = brewer.pal(4, "OrRd")[4]

require(ggplot2)
ggplot(d, aes(x = x1, y = value, group=variable, color=col)) + 
  geom_line()+
  geom_ribbon(aes(ymin=d_low,ymax=d_high),alpha=0.3)  +
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
  guides(fill = guide_legend(keywidth = 0.5, keyheight = 0.5, nrow = 2))

ggsave("vel.pdf", height = 8, width = 8, units = "cm")
