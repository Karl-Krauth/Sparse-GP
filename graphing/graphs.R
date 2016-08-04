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
h = 7
output_path = "./"

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

# sarcos all joints inducing data
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
