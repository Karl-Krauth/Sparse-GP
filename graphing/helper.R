rename_model <- function(data){
  data$model[data$model == 'MIX1'] = "MoG1"
  data$model[data$model == 'MIX2'] = "MoG2"
  data$model[data$model == 'FULL'] = "FG"
  data
}

draw_bar_models <- function(data, y_lab, leg_pos){
  data = melt(data)
  data$model = toupper(substr(data$variable,0, 4))
  data = rename_model(data)
  data$sp = substr(data$variable,6, 15)
  
  ggplot(data, aes(x="", y = value, fill = sp)) + 
    stat_summary(fun.y = "mean", geom = "bar", position = position_dodge() ) + 
    stat_summary(fun.data = mean_cl_normal, geom="linerange", colour="black", position=position_dodge(.9)) +
    theme_bw() + 
    scale_fill_brewer(name=SP_name, palette="Set1") + 
    
    xlab('') +
    ylab(y_lab) +
    theme(legend.direction = "vertical", legend.position = leg_pos, legend.box = "vertical", 
          axis.line = element_line(colour = "black"),
          panel.grid.major=element_blank(), 
          panel.grid.minor=element_blank(),      
          panel.border = element_blank(),
          panel.margin = unit(.4, "lines"),
          text=element_text(family="Arial", size=10),
          legend.key = element_blank(),
          strip.background = element_rect(colour = "white", fill = "white",
                                          size = 0.5, linetype = "solid"),
          axis.ticks.x = element_blank(),
          axis.title.x=element_blank()
          
          
    ) +
    facet_wrap(~model)+ 
    
    guides(fill = guide_legend(keywidth = 0.5, keyheight = 0.5))
}  

draw_bar_models_with_X <- function(data, y_lab, leg_pos){
  data = melt(data)
  data$model = toupper(substr(data$variable,0, 4))
  data = rename_model(data)
  data$sp = substr(data$variable,6, 15)
  
  ggplot(data, aes(x=sp, y = value, fill = sp)) + 
    stat_summary(fun.y = "mean", geom = "bar", position = position_dodge() ) + 
    stat_summary(fun.data = mean_cl_normal, geom="linerange", colour="black", position=position_dodge(.9)) +
    theme_bw() + 
    scale_fill_brewer(name=SP_name, palette="Set1") + 
    xlab(SP_name) +
    ylab(y_lab) +
    theme(legend.direction = "vertical", legend.position = leg_pos, legend.box = "vertical", 
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
    facet_wrap(~model)+ 
    
    guides(fill = guide_legend(keywidth = 0.5, keyheight = 0.5))
}  



draw_boxplot_models_with_X <- function(data, y_lab, leg_pos){
  data = melt(data)
  data$model = toupper(substr(data$variable,0, 4))
  data$sp = substr(data$variable,6, 15)
  data = rename_model(data)
  #y_max = max(by(data, data[, c('model', 'sp')], function(x){quantile(x$value, 0.975)}))
  #y_min = min(data$value)
  y_max = max(by(data, data[, c('model', 'sp')], function(x){boxplot.stats(x$value)$stats[c(5)]}))
  y_min = min(by(data, data[, c('model', 'sp')], function(x){boxplot.stats(x$value)$stats[c(1)]}))
  p = ggplot(data, aes(x=sp, y = value, colour = sp)) + 
    geom_boxplot(width=1, 
                 position=position_dodge(1),
                 outlier.shape = NA) + 
    coord_cartesian(ylim = c(y_min - abs(y_min) * 0.1 , y_max + abs(y_max) * 0.1)) +
    theme_bw() + 
    scale_colour_brewer(name=SP_name, palette="Set1") +
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
          legend.key = element_blank(),
          strip.background = element_rect(colour = "white", fill = "white",
                                          size = 0.5, linetype = "solid")
    ) +
    facet_wrap(~model)+ 
    
    guides(fill = guide_legend(keywidth = 0.5, keyheight = 0.5))
  p
}

draw_intensity <- function(data, y_lab){
  p = ggplot(data, aes(x=x, y = m, colour = sp)) + 
    geom_line() +
    geom_ribbon(aes(x=x, ymin= m - 2 * sqrt(v), ymax=m + 2 * sqrt(v)), fill="grey", alpha=.4, colour =NA) +  
    scale_colour_brewer(palette="Set1") +
    xlab('') +
    ylab(y_lab) +
    theme_bw() + 
    
    theme(legend.direction = "vertical", legend.position = "none", legend.box = "vertical", 
          axis.line = element_line(colour = "black"),
          panel.grid.minor=element_blank(),      
          panel.border = element_blank(),
          panel.margin = unit(.4, "lines"),
          text=element_text(family="Arial", size=10),
          legend.key = element_blank(),
          strip.background = element_rect(colour = "white", fill = "white",
                                          size = 0.5, linetype = "solid"),
          axis.ticks.x = element_blank(),
          legend.title=element_blank(),
          axis.text.x = element_text(angle = 90, hjust = 1)
    ) +
    facet_grid(model ~ sp)+ 
    guides(fill = guide_legend(keywidth = 0.5, keyheight = 0.5))
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
