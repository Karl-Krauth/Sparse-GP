rename_model <- function(data){
  data$model[data$model == 'MIX1'] = "MoG1"
  data$model[data$model == 'MIX2'] = "MoG2"
  data$model[data$model == 'FULL'] = "FG"
  data$model[data$model == 'FULI'] = "IND"
  data$model[data$model == 'BASE'] = "GP"
  data$model[data$model == 'GAUS'] = "GP"
  data$model[data$model == 'WARP'] = "WGP"
  data$model[data$model == 'EXPP'] = "EP"
  data$model[data$model == 'VARB'] = "VBO"
  data$model[data$model == 'EMTB'] = "VQ"
  data$model[data$model == 'ELSS'] = "ESS"
  data$model[data$model == 'HMOC'] = "HMC"
  data
}

draw_bar_models_with_X <- function(data, y_lab, leg_pos){
  data = melt(data)
  data$model = toupper(substr(data$variable,0, 4))
  data = rename_model(data)
  data$sp = substr(data$variable,6, 15)
  data$sp = factor(data$sp, levels=c("0.001", "0.004", "0.02", "0.04", "0.1", "0.2", "0.5", "1.0", ""))
  data$model2 = factor(data$model, levels=c("FG", "IND", "MoG1", "MoG2", "GP", "WGP", "EP", "VBO", "VQ"))

  ggplot(data, aes(x=sp, y = value, fill = sp)) +
    stat_summary(fun.y = "mean", geom = "bar", position = position_dodge(),color="black", size=0.2) +
    stat_summary(fun.data = mean_cl_normal, geom="linerange", colour="black", position=position_dodge(.9)) +
    theme_bw() +
    scale_fill_brewer(name=SP_name, palette="OrRd") +
    xlab(SP_name) +
    ylab(y_lab) +
    theme(legend.direction = "vertical", legend.position = leg_pos, legend.box = "vertical",
          axis.line.x = element_line(colour = "black"),
          axis.line.y = element_line(colour = "black"),
          panel.grid.major=element_blank(),
          panel.grid.minor=element_blank(),
          panel.border = element_blank(),
          panel.margin = unit(.4, "lines"),
          text=element_text(family="Arial", size=10),
          legend.key = element_blank()
    ) +
    facet_grid(~model2, scales="free_x", space="free_x") +

    guides(fill = guide_legend(keywidth = 0.5, keyheight = 0.5))
}



draw_boxplot_models_with_X <- function(data, y_lab, leg_pos){
  data = melt(data)
  data$model = toupper(substr(data$variable,0, 4))
  data$sp = substr(data$variable,6, 15)
  data = rename_model(data)
  #y_max = max(by(data, data[, c('model', 'sp')], function(x){quantile(x$value, 0.975)}))
  #y_min = min(data$value)
  y_max = by(data, data[, c('model', 'sp')], function(x){boxplot.stats(x$value)$stats[c(5)]})
  y_max = max(y_max[!is.na(y_max)])
  y_min = by(data, data[, c('model', 'sp')], function(x){boxplot.stats(x$value)$stats[c(1)]})
  y_min = min(y_min[!is.na(y_min)])
  data$sp = factor(data$sp, levels=c("0.001", "0.004", "0.02", "0.04", "0.1", "0.2", "0.5", "1.0", ""))
  data$model2 = factor(data$model, levels=c("FG", "IND", "MoG1", "MoG2", "GP", "WGP", "EP", "VBO", "VQ"))
  p = ggplot(data, aes(x=sp, y = value, fill = sp)) +
    geom_boxplot(width=1,
                 position=position_dodge(1),
                 outlier.shape = NA) +
    coord_cartesian(ylim = c(y_min - abs(y_min) * 0.1 , y_max + abs(y_max) * 0.1)) +
    theme_bw() +
    scale_fill_brewer(palette="OrRd") +
    theme_bw() +
    xlab(SP_name) +
    ylab(y_lab) +
    theme(legend.direction = "vertical", legend.position = "none", legend.box = "vertical",
          axis.line.x = element_line(colour = "black", size=0.5),
          axis.line.y = element_line(colour = "black", size=0.5),
          panel.grid.major=element_blank(),
          panel.grid.minor=element_blank(),
          panel.margin = unit(.4, "lines"),
          text=element_text(family="Arial", size=10),
          panel.border = element_rect(color = "black", fill = NA, size = 0.0),
          legend.key = element_blank()
    ) +
    facet_grid(~model2, scales="free_x", space="free_x") +
    guides(fill = guide_legend(keywidth = 0.5, keyheight = 0.5))
  p
}

get_quadrant <- function(m, v, alpha){
  # Compute exp(fmu + sqrt(fvar) * alpha) * offset in the transformed space.
  offset = 191.0 / 811.0
  m = m / offset
  v = v / (offset ** 2)
  res = (m ** 2 / sqrt(v + m ** 2)) * exp(alpha * sqrt(log((v + m ** 2) / m ** 2))) * offset
  res
}

draw_intensity <- function(data, y_lab){
  hmc = data[data$model == "HMC",][c("x", "m", "v")]
  ess = data[data$model == "ESS",][c("x", "m", "v")]
  data = data[data$model != "HMC" & data$model != "ESS",]
  data$model2 = factor(data$model, levels=c("FG", "MoG1", "MoG2", "GP", "WGP", "EP", "VBO", "VQ", "HMC", "ESS"))
  p = ggplot(data, aes(x=x, y = m, colour = sp)) +
    geom_line() +
    geom_ribbon(aes(x=x, ymin=get_quadrant(m, v, -1.96), ymax=get_quadrant(m, v, 1.96)),
                fill="grey", alpha=.4, colour =NA) +
    scale_colour_brewer(palette="Set1") +
    xlab('') +
    ylab(y_lab) +
    theme_bw() +

    theme(legend.direction = "vertical", legend.position = "none", legend.box = "vertical",
          axis.line = element_line(color = "black"),
          panel.grid.minor=element_blank(),
          panel.border = element_blank(),
          axis.line.x = element_line(color="black", size = 0.5),
          axis.line.y = element_line(color="black", size = 0.5),
          panel.margin = unit(.4, "lines"),
          text=element_text(family="Arial", size=10),
          legend.key = element_blank(),
          axis.ticks.x = element_blank(),
          legend.title=element_blank(),
          axis.text.x = element_text(angle = 90, hjust = 1)
    ) +
    facet_grid(model2 ~ sp) +
    guides(fill = guide_legend(keywidth = 0.5, keyheight = 0.5)) +
    geom_line(data=hmc, aes(x=x, y=m), colour="green", linetype="dashed") +
    geom_line(data=hmc, aes(x=x, y=get_quadrant(m, v, 1.96)), colour="green", linetype="dashed") +
    geom_line(data=hmc, aes(x=x, y=get_quadrant(m, v, -1.96)), colour="green", linetype="dashed") +
    geom_line(data=hmc, aes(x=x, y=m), colour="orange", linetype="dotted") +
    geom_line(data=hmc, aes(x=x, y=get_quadrant(m, v, 1.96)), colour="orange", linetype="dotted") +
    geom_line(data=hmc, aes(x=x, y=get_quadrant(m, v, -1.96)), colour="orange", linetype="dotted")
  p
}

draw_mining_data <- function(data){
  p = ggplot(data, aes(x=x, y = y)) +
    stat_summary(fun.y = "mean", geom = "line", position = position_dodge()) +


    theme_bw() +
    scale_colour_brewer(palette="Set1") +
    xlab('time') +
    ylab('event counts') +
    theme(legend.direction = "vertical", legend.position = "none", legend.box = "vertical",
          axis.line = element_line(colour = "black"),
          panel.grid.major=element_blank(),
          panel.grid.minor=element_blank(),
          panel.border = element_blank(),
          axis.line.x = element_line(color="black", size = 0.5),
          axis.line.y = element_line(color="black", size = 0.5),
          panel.margin = unit(.4, "lines"),
          text=element_text(family="Arial", size=10),
          legend.key = element_blank(),
          strip.background = element_rect(colour = "white", fill = "white",
                                          size = 0.5, linetype = "solid")
    ) +
    guides(fill = guide_legend(keywidth = 0.5, keyheight = 0.5))
  p
}

draw_joints <- function(data){
  data = melt(data)
  data$joint = factor(as.numeric(substr(data$variable,11, 14)) + 1)
  data$name = paste(SP_name, "=", "0.04")
  p =   ggplot(data, aes(x=joint, y = value)) +
    stat_summary(fun.y = "mean", geom = "bar", fill="gray", colour = "black",position = position_dodge() ) +
    theme_bw() +

    xlab("output") +
    ylab("SMSE") +
    theme(legend.direction = "vertical", legend.position = "none", legend.box = "vertical",
          axis.line = element_line(colour = "black"),
          panel.grid.major=element_blank(),
          panel.grid.minor=element_blank(),
          panel.border = element_blank(),
          panel.margin = unit(.4, "lines"),
          text=element_text(family="Arial", size=10),
          legend.key = element_blank(),
          strip.background = element_rect(colour = "white", fill = "white",
                                          size = 0.5, linetype = "solid")


    ) +
    facet_grid(. ~ name)+


    guides(fill = guide_legend(keywidth = 0.5, keyheight = 0.5))
  p
}
