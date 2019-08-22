input_path = "seismic_24-May-2018_16h32m05s_17885/" 
fname = paste0(input_path, "elbo.csv")
d  = read.csv(fname)
ggplot(d, aes(x=iter, y=nelbo)) + geom_line()
