library(languageR)
library(ggplot2)
library(plotrix)
library(plyr)
library(party)
library(gridExtra)
library(Hmisc)
library(extrafont)
library(scales)
library(reshape)
library(pbkrtest)
library(nloptr)
library(optimx)
library(data.table)
library(extrafont)
loadfonts()

#some helper functions
source('helper.R')

SP_name = "SF"

w = 12
h = 4.5
output_path = "./"

merge_figs = function(p1, p2)
{
  grid.newpage()


# extract gtable
  g1 <- ggplot_gtable(ggplot_build(p1))
  g2 <- ggplot_gtable(ggplot_build(p2))
  
  # overlap the panel of 2nd plot on that of 1st plot
  pp <- c(subset(g1$layout, name == "panel", se = t:r))
  g <- gtable_add_grob(g1, g2$grobs[[which(g2$layout$name == "panel")]], pp$t, 
                       pp$l, pp$b, pp$l)
  
  # axis tweaks
  ia <- which(g2$layout$name == "axis-l")
  ga <- g2$grobs[[ia]]
  ax <- ga$children[[2]]
  ax$widths <- rev(ax$widths)
  ax$grobs <- rev(ax$grobs)
  g <- gtable_add_cols(g, g2$widths[g2$layout[ia, ]$l], length(g$widths) - 1)
  g <- gtable_add_grob(g, ax, pp$t, length(g$widths) - 1, pp$b)
  
  # draw it
  p1 = grid.draw(g)
  g
}

#if (FALSE) {
# boston data
data = read.csv('../../graph_data/boston_SSE_data.csv')
p1 = draw_boxplot_models_with_X(data, "SSE", "None")
ggsave(file=paste(output_path, "boston_SSE", ".pdf", sep = ""),  width=w, height=h, units = "cm" , device=cairo_pdf, p1)

data = read.csv('../../graph_data/boston_NLPD_data.csv')
p2 = draw_boxplot_models_with_X(data, "NLPD", "right")
ggsave(file=paste(output_path, "boston_NLPD", ".pdf", sep = ""),  width=w, height=h, units = "cm" , device=cairo_pdf, p2)

# abalone data
data = read.csv('../../graph_data/abalone_SSE_data.csv')
p1 = draw_boxplot_models_with_X(data, "SSE", "None")
ggsave(file=paste(output_path, "abalone_SSE", ".pdf", sep = ""),  width=w, height=h, units = "cm" , device=cairo_pdf, p1)

data = read.csv('../../graph_data/abalone_NLPD_data.csv')
p2 = draw_boxplot_models_with_X(data, "NLPD", "right")
ggsave(file=paste(output_path, "abalone_NLPD", ".pdf", sep = ""),  width=w, height=h, units = "cm" , device=cairo_pdf, p2)


# creep data
data = read.csv('../../graph_data/creep_SSE_data.csv')
p1 = draw_boxplot_models_with_X(data, "SSE", "None")
ggsave(file=paste(output_path, "creep_SSE", ".pdf", sep = ""),  width=w, height=h, units = "cm" , device=cairo_pdf, p1)

data = read.csv('../../graph_data/creep_NLPD_data.csv')
p2 = draw_boxplot_models_with_X(data, "NLPD", "right")
ggsave(file=paste(output_path, "creep_NLPD", ".pdf", sep = ""),  width=w, height=h, units = "cm" , device=cairo_pdf, p2)

# mining data
data = read.csv('../../graph_data/mining_true_y_data.csv')
p1 = draw_mining_data(data)
ggsave(file=paste(output_path, "mining_EVENTS", ".pdf", sep = ""),  width=w, height=h, units = "cm" , device=cairo_pdf, p1)

data = read.csv('../../graph_data/mining_intensity_data.csv')
data$model = toupper(substr(data$model_sp,0, 4))
data = rename_model(data)
data$sp = paste(SP_name, "=", substr(data$model_sp,6, 8))
p2 = draw_intensity(data, "intensity")
ggsave(file=paste(output_path, "mining_INTENSITY", ".pdf", sep = ""), width=w,  height=2*h, units = "cm" , device=cairo_pdf, p2)

# wisc data ####
data = read.csv('../../graph_data/breast_cancer_ER_data.csv')
p1 = draw_bar_models_with_X(data, "error rate", "None")
ggsave(file=paste(output_path, "breast_cancer_ER", ".pdf", sep = ""),  width=w, height=h, units = "cm" , device=cairo_pdf, p1)

data = read.csv('../../graph_data/breast_cancer_NLPD_data.csv')
p2 = draw_boxplot_models_with_X(data, "NLP",  "right")
ggsave(file=paste(output_path, "breast_cancer_NLPD", ".pdf", sep = ""),  width=w, height=h, units = "cm" , device=cairo_pdf, p2)

#usps data
data = read.csv('../../graph_data/usps_ER_data.csv')
p1 = draw_bar_models_with_X(data, "error rate", "None")
ggsave(file=paste(output_path, "usps_ER", ".pdf", sep = ""),  width=w, height=h, units = "cm" , device=cairo_pdf, p1)

data = read.csv('../../graph_data/usps_NLPD_data.csv')
p2 = draw_boxplot_models_with_X(data, "NLP", "right")
ggsave(file=paste(output_path, "usps_NLPD", ".pdf", sep = ""),  width=w, height=h, units = "cm" , device=cairo_pdf, p2)

#mnist binary data
data = read.csv('../../graph_data/mnist_binary_ER_data.csv')
p1 = draw_bar_models_with_X(data, "error rate", "None")
ggsave(file=paste(output_path, 'mnist_binary_ER', ".pdf", sep = ""),  width=w/2, height=h, units = "cm" , device=cairo_pdf, p1)

data = read.csv('../../graph_data/mnist_binary_NLPD_data.csv')
p2 = draw_boxplot_models_with_X(data, "NLP", "None")
ggsave(file=paste(output_path, 'mnist_binary_NLPD', ".pdf", sep = ""),  width=w/2, height=h, units = "cm" , device=cairo_pdf, p2)

# mnist binary inducing data
inducing_data = read.csv('../../graph_data/mnist_binary_inducing_ER_data.csv')
standard_data = read.csv('../../graph_data/mnist_binary_ER_data.csv')
standard_data = standard_data[, -grep("0\\.00[14]", colnames(standard_data))]
data = cbind(standard_data, inducing_data)
p1 = draw_bar_models_with_X(data, "error rate", "None")
ggsave(file=paste(output_path, 'mnist_binary_inducing_ER', ".pdf", sep = ""),  width=w/2, height=h, units = "cm" , device=cairo_pdf, p1)

inducing_data = read.csv('../../graph_data/mnist_binary_inducing_NLPD_data.csv')
standard_data = read.csv('../../graph_data/mnist_binary_NLPD_data.csv')
standard_data = standard_data[, -grep("0\\.00[14]", colnames(standard_data))]
data = cbind(standard_data, inducing_data)
p2 = draw_boxplot_models_with_X(data, "NLP", "None")
ggsave(file=paste(output_path, 'mnist_binary_inducing_NLPD', ".pdf", sep = ""),  width=w/2, height=h, units = "cm" , device=cairo_pdf, p2)

#mnist data
data = read.csv('../../graph_data/mnist_ER_data.csv')
p1 = draw_bar_models_with_X(data, "error rate", "None")
ggsave(file=paste(output_path, 'mnist_ER', ".pdf", sep = ""),  width=w/2, height=h, units = "cm" , device=cairo_pdf, p1)

data = read.csv('../../graph_data/mnist_NLPD_data.csv')
p2 = draw_boxplot_models_with_X(data, "NLP", "None")
ggsave(file=paste(output_path, 'mnist_NLPD', ".pdf", sep = ""),  width=w/2, height=h, units = "cm" , device=cairo_pdf, p2)

# mnist inducing data
inducing_data = read.csv('../../graph_data/mnist_inducing_ER_data.csv')
standard_data = read.csv('../../graph_data/mnist_ER_data.csv')
standard_data = standard_data[, -grep("0\\.00[14]", colnames(standard_data))]
data = cbind(standard_data, inducing_data)
p1 = draw_bar_models_with_X(data, "error rate", "None")
ggsave(file=paste(output_path, 'mnist_inducing_ER', ".pdf", sep = ""),  width=w/2, height=h, units = "cm" , device=cairo_pdf, p1)

inducing_data = read.csv('../../graph_data/mnist_inducing_NLPD_data.csv')
standard_data = read.csv('../../graph_data/mnist_NLPD_data.csv')
standard_data = standard_data[, -grep("0\\.00[14]", colnames(standard_data))]
data = cbind(standard_data, inducing_data)
p2 = draw_boxplot_models_with_X(data, "NLP", "None")
ggsave(file=paste(output_path, 'mnist_inducing_NLPD', ".pdf", sep = ""),  width=w/2, height=h, units = "cm" , device=cairo_pdf, p2)

#sarcos data
data = read.csv('../../graph_data/sarcos_MSSE_data.csv')
p1 = draw_bar_models_with_X(data, "MSSE", "None")
ggsave(file=paste(output_path, 'sarcos_MSSE', ".pdf", sep = ""),  width=w/2, height=h, units = "cm" , device=cairo_pdf, p1)

data = read.csv('../../graph_data/sarcos_NLPD_data.csv')
p2 = draw_boxplot_models_with_X(data, "NLPD", "None")
ggsave(file=paste(output_path, 'sarcos_NLPD', ".pdf", sep = ""),  width=w/2, height=h, units = "cm" , device=cairo_pdf, p2)

#sarcos all joints data
data = read.csv('../../graph_data/sarcos_all_joints_MSSE_data.csv')
p1 = draw_bar_models_with_X(data, "MSSE", "None")
ggsave(file=paste(output_path, 'sarcos_all_joints_MSSE', ".pdf", sep = ""),  width=w/2, height=h, units = "cm" , device=cairo_pdf, p1)

data = read.csv('../../graph_data/sarcos_all_joints_NLPD_data.csv')
p2 = draw_boxplot_models_with_X(data, "NLPD", "None")
ggsave(file=paste(output_path, 'sarcos_all_joints_NLPD', ".pdf", sep = ""),  width=w/2, height=h, units = "cm" , device=cairo_pdf, p2)

# sarcos inducing data
inducing_data = read.csv('../../graph_data/sarcos_MSSE_data_inducing.csv')
standard_data = read.csv('../../graph_data/sarcos_MSSE_data.csv')
standard_data = standard_data[, -grep("0\\.00[14]", colnames(standard_data))]
data = cbind(standard_data, inducing_data)
p1 = draw_bar_models_with_X(data, "MSSE", "None")
ggsave(file=paste(output_path, 'sarcos_inducing_MSSE', ".pdf", sep = ""),  width=w/2, height=h, units = "cm" , device=cairo_pdf, p1)

inducing_data = read.csv('../../graph_data/sarcos_NLPD_data_inducing.csv')
standard_data = read.csv('../../graph_data/sarcos_NLPD_data.csv')
standard_data = standard_data[, -grep("0\\.00[14]", colnames(standard_data))]
data = cbind(standard_data, inducing_data)
p2 = draw_boxplot_models_with_X(data, "NLPD", "None")
ggsave(file=paste(output_path, 'sarcos_inducing_NLPD', ".pdf", sep = ""),  width=w/2, height=h, units = "cm" , device=cairo_pdf, p2)

# sarcosi all joints inducing data
inducing_data = read.csv('../../graph_data/sarcos_all_joints_MSSE_data_inducing.csv')
standard_data = read.csv('../../graph_data/sarcos_all_joints_MSSE_data.csv')
standard_data = standard_data[, -grep("0\\.00[14]", colnames(standard_data))]
data = cbind(standard_data, inducing_data)
p1 = draw_bar_models_with_X(data, "MSSE", "None")
ggsave(file=paste(output_path, 'sarcos_all_joints_inducing_MSSE', ".pdf", sep = ""),  width=w/2, height=h, units = "cm" , device=cairo_pdf, p1)

inducing_data = read.csv('../../graph_data/sarcos_all_joints_NLPD_data_inducing.csv')
standard_data = read.csv('../../graph_data/sarcos_all_joints_NLPD_data.csv')
standard_data = standard_data[, -grep("0\\.00[14]", colnames(standard_data))]
data = cbind(standard_data, inducing_data)
p2 = draw_boxplot_models_with_X(data, "NLPD", "None")
ggsave(file=paste(output_path, 'sarcos_all_joints_inducing_NLPD', ".pdf", sep = ""),  width=w/2, height=h, units = "cm" , device=cairo_pdf, p2)

#for ailine data
lin_RMSE = c(38.5742766187, 40.8273352476, 34.578759036, 36.7876864506, 45.3542697617)

lin_NLPD = c(5.08685873604, 5.13457483244, 4.97758451918, 5.02466269926, 5.27594624659)


output = list()
index = 1
for(i in 0:4){
  nlpd = read.csv(paste("../../graph_data/airline/airline", i, "/nlpd.txt", sep=""), header = F)
  rmse = read.csv(paste("../../graph_data/airline/airline", i, "/rmse.txt", sep=""), header = F)
  obj = read.csv(paste("../../graph_data/airline/airline", i, "/obj.txt", sep=""), header = F)
  output[[index]] = data.frame(nlpd = as.vector(nlpd[,1]), rmse = as.vector(rmse[,1]), obj = as.vector(obj[,1]),
                               expr = i, 
                               method="SAVIGP", epoch=1:nrow(nlpd))
  index = index + 1
  nlpd = read.csv(paste("../../graph_data/airline/airline", i, "/nlpdsvi.txt", sep=""), header = F)
  rmse = read.csv(paste("../../graph_data/airline/airline", i, "/rmsesvi.txt", sep=""), header = F)
  output[[index]] = data.frame(nlpd = as.vector(nlpd[,1]), rmse = as.vector(rmse[,1]), obj=NA, expr = i, 
                               method="SVI", epoch=1:nrow(nlpd))
  total_epochs = nrow(nlpd)
  index = index + 1
  
  nlpd = read.csv(paste("../../graph_data/airline/airline", i, "/nlpdlargegp.txt", sep=""), header = F)
  rmse = read.csv(paste("../../graph_data/airline/airline", i, "/rmselargegp.txt", sep=""), header = F)
  output[[index]] = data.frame(nlpd = mean(nlpd[,1]), rmse = mean(rmse[,1]), obj=NA, expr = i, 
                               method="GP2000", epoch=1:total_epochs)
  index = index + 1
  
  nlpd = read.csv(paste("../../graph_data/airline/airline", i, "/nlpdsmallgp.txt", sep=""), header = F)
  rmse = read.csv(paste("../../graph_data/airline/airline", i, "/rmsesmallgp.txt", sep=""), header = F)
  output[[index]] = data.frame(nlpd = mean(nlpd[,1]), rmse = mean(rmse[,1]), obj=NA, expr = i, 
                               method="GP1000", epoch=1:total_epochs)
  index = index + 1
  
  output[[index]] = data.frame(nlpd = lin_NLPD[i+1], rmse = lin_RMSE[i+1], obj=NA, expr = i, 
                               method="LINEAR", epoch=1:total_epochs)
  index = index + 1
  
}
require(data.table)
res = rbindlist(output)

require(ggplot2)
draw_line_chart = function(res, ylabel, legend="none") {
  p1 = ggplot(res, aes(x=epoch, y = y, color = method)) +
  stat_summary(fun.y = "mean", geom="line", size=1) +
  theme_bw() +
  scale_color_brewer(palette="Set1", name = "") +
  theme_bw() +
  xlab("epoch") +
  ylab(ylabel) +
  scale_x_continuous(expand = c(0, 0)) + 
  theme(legend.direction = "vertical", legend.box = "vertical", legend.position=legend,
        axis.line.x = element_line(colour = "black", size=0.5),
        axis.line.y = element_line(colour = "black", size=0.5),
        panel.grid.major=element_blank(),
        panel.grid.minor=element_blank(),
        text=element_text(family="Arial", size=10),
        panel.border = element_rect(color = "black", fill = NA, size = 0.0),
        legend.key = element_blank(),
        panel.background = element_rect(fill = NA)
  )+
  guides(fill = guide_legend(keywidth = 0.5, keyheight = 0.5))
  p1
}



res$y = res$nlpd
p1 = draw_line_chart(res, "NLPD", legend = "right")
ggsave(file=paste(output_path, 'airline_NLPD', ".pdf", sep = ""),  
       width=w/1.5, height=h, units = "cm" , device=cairo_pdf, p1)

res$y = res$rmse
p1 = draw_line_chart(res, "RMSE", legend = "none")
ggsave(file=paste(output_path, 'airline_RMSE', ".pdf", sep = ""),  
       width=w/2, height=h, units = "cm" , device=cairo_pdf, p1)

library(gtable)
library(grid)

draw_line_chart2 = function(res, ylabel, color, name) {
  p1 = ggplot(res, aes(x=epoch, y = y, color=method)) +
    stat_summary(fun.y = "mean", geom="line", size=1, color=color) +
    theme_bw() +
    scale_color_brewer(palette="Set1", name = "") +
    theme_bw() +
    xlab("epoch") +
    ylab(ylabel) +
    scale_x_continuous(expand = c(0, 0)) + 
    theme_bw()+
    theme(legend.direction = "vertical", legend.box = "vertical",legend.position = c(0.8, 0.2),
          axis.line.x = element_line(colour = "black", size=0.5),
          axis.line.y = element_line(colour = "black", size=0.5),
          panel.grid.major=element_blank(),
          panel.grid.minor=element_blank(),
          text=element_text(family="Arial", size=10),
          panel.border = element_rect(color = "black", fill = NA, size = 0.5),
          legend.key = element_blank(),
          panel.background = element_rect(fill = NA),
          axis.ticks.y = element_blank()
    )
  p1
}


res$y = res$obj
p1 = draw_line_chart2(subset(res, method=="SAVIGP"), "NELBO", "blue")

res$y = res$rmse
p2 = draw_line_chart2(subset(res, method=="SAVIGP"), "RMSE", "red")

g = merge_figs(p1, p2)
ggsave(file=paste(output_path, 'airline_NELBO_RMSE', ".pdf", sep = ""),  
       width=w/1.7, height=h, units = "cm" , device=cairo_pdf, g)


res$y = res$obj
p1 = draw_line_chart2(subset(res, method=="SAVIGP"), "NELBO", "blue")

res$y = res$nlpd
p2 = draw_line_chart2(subset(res, method=="SAVIGP"), "NLPD", "red")

g = merge_figs(p1, p2)
ggsave(file=paste(output_path, 'airline_NELBO_NLPD', ".pdf", sep = ""),  
       width=w/1.7, height=h, units = "cm" , device=cairo_pdf, g)


