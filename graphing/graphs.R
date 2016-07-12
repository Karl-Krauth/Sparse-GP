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

w = 15
h = 5
output_path = "../../SAVIGP_paper/nips2015/figures/raw/"

# boston data
name= 'boston'
data = read.csv('../../graph_data/boston_SSE_data.csv')
p1 = draw_boxplot_models_with_X(data, "SSE", "None")

data = read.csv('../../graph_data/boston_NLPD_data.csv')
p2 = draw_boxplot_models_with_X(data, "NLPD", "right")

g = arrangeGrob(p1, p2, ncol=2,  widths=c(9/20, 11/20))
ggsave(file=paste(output_path, name, ".pdf", sep = ""),  width=w, height=h, units = "cm" , device=cairo_pdf, g)      

# abalone data
name= 'abalone'
data = read.csv('../../graph_data/abalone_SSE_data.csv')
p1 = draw_boxplot_models_with_X(data, "SSE", "None")

data = read.csv('../../graph_data/abalone_NLPD_data.csv')
p2 = draw_boxplot_models_with_X(data, "NLPD", "right")

g = arrangeGrob(p1, p2, ncol=2,  widths=c(9/20, 11/20))
ggsave(file=paste(output_path, name, ".pdf", sep = ""),  width=w, height=h, units = "cm" , device=cairo_pdf, g)      


# creep data
name = "creep"
data = read.csv('../../graph_data/creep_SSE_data.csv')
p1 = draw_boxplot_models_with_X(data, "SSE", "None")

data = read.csv('../../graph_data/creep_NLPD_data.csv')
p2 = draw_boxplot_models_with_X(data, "NLPD", "right")

g = arrangeGrob(p1, p2, ncol=2,  widths=c(9/20, 11/20))
ggsave(file=paste(output_path, name, ".pdf", sep = ""),  width=w, height=h, units = "cm" , device=cairo_pdf, g)      


# mining data
name = 'mining'
data = read.csv('../../graph_data/mining_intensity_data.csv')
data$model = toupper(substr(data$model_sp,0, 4))
data = rename_model(data)
data$sp = paste(SP_name, "=", substr(data$model_sp,6, 8))
p2 = draw_intensity(data, "intensity")

data = read.csv('../../graph_data/mining_true_y_data.csv')
p1 = draw_mining_data(data)
g = arrangeGrob(p1, p2, ncol=2,  widths=c(8/20, 12/20))
ggsave(file=paste(output_path, name, ".pdf", sep = ""),  width=w, height=h, units = "cm" , device=cairo_pdf, g)      



# wisc data ####
name = "wisc"
data = read.csv('../../graph_data/wisc_ER_data.csv')
p1 = draw_bar_models(data, "error rate", "None")

data = read.csv('../../graph_data/wisc_NLPD_data.csv')
p2 = draw_boxplot_models_with_X(data, "NLP",  "right")

g = arrangeGrob(p1, p2, ncol=2,  widths=c(9/20, 11/20))
ggsave(file=paste(output_path, name, ".pdf", sep = ""),  width=w, height=h, units = "cm" , device=cairo_pdf, g)      


#usps data
name = "usps"
data = read.csv('../../graph_data/usps_ER_data.csv')
p1 = draw_bar_models(data, "error rate", "None")

data = read.csv('../../graph_data/usps_NLPD_data.csv')
p2 = draw_boxplot_models_with_X(data, "NLP", "right")

g = arrangeGrob(p1, p2, ncol=2,  widths=c(9/20, 11/20))
ggsave(file=paste(output_path, name, ".pdf", sep = ""),  width=w, height=h, units = "cm" , device=cairo_pdf, g)      


#mnist data
name = "mnist_sarcos"
data = read.csv('../../graph_data/mnist_ER_data.csv')
p1 = draw_bar_models_with_X(data, "error rate", "None")

data = read.csv('../../graph_data/mnist_NLPD_data.csv')
p2 = draw_boxplot_models_with_X(data, "NLP", "None")


data = read.csv('../../graph_data/sarcos_MSSE_data.csv')
p3 = draw_joints(data)

g = arrangeGrob(p1, p2, p3, ncol=3,  widths=c(11/30, 11/30, 8/30))
ggsave(file=paste(output_path, name, ".pdf", sep = ""),  width=w, height=h, units = "cm" , device=cairo_pdf, g)      
