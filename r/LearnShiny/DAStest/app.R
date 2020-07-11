#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#

library(shiny)
knitr::opts_chunk$set(echo = TRUE)
library(shinyWidgets)
library(plotly)
library(ggthemes)
N = 35
set.seed(0)
index = sample(x = 1:N,
               size = N,
               replace = FALSE)
possible_choices <- c('Agree Strongly' = -2,
                      'Agree Slightly'=-1,
                      'Neutral'=0,
                      'Disagree Slightly'=1,
                      'Disagree Very Much'=2
)
origin_ques = readLines('data/raw.txt')
for(i in 1:N){
    origin_ques[i] = unlist(strsplit(origin_ques[i],'[.]'))[2]
}
ques = origin_ques[index]
ans = rep(0.0,N)
origin_ans = rep(0.0,N)

ValueSystems = c("Approval","Love","Achievement","Perfectionism",
                 "Entitlement","Omnipotence","Autonomy")
score = rep(0.0,length(ValueSystems))
names(score) = ValueSystems
VS_index = c(rep(ValueSystems[1],5),
             rep(ValueSystems[2],5),
             rep(ValueSystems[3],5),
             rep(ValueSystems[4],5),
             rep(ValueSystems[5],5),
             rep(ValueSystems[6],5),
             rep(ValueSystems[7],5))


# Define UI for application that draws a histogram
ui <- fluidPage(

    # Application title
    titlePanel("Simple DAS Test"),

    # Sidebar with a slider input for number of bins 
    lapply(1:N, function(i) {
        radioGroupButtons(inputId = paste("q", i, sep = ''),
                          label = paste(i,'. ', ques[i],sep = ''), 
                          selected = NULL,
                          choices = possible_choices,
                          width = '300%')
    }),
        # Show a plot of the generated distribution
    tableOutput('table'),
    plotOutput('radar')
    )


# Define server logic required to draw a histogram
server <- function(input, output) {
    output$table = renderTable({
        for(i in 1:N){
            ans[i] <- input[[paste("q",i,sep = '')]]
            origin_ans[index[i]] <- ans[i]
            score[VS_index[index[i]]] = score[VS_index[index[i]]] + as.numeric(ans[i])
        }
        res = as.data.frame(t(score))
    })
    output$radar = renderPlot(
        {
            for(i in 1:N){
                ans[i] <- input[[paste("q",i,sep = '')]]
                origin_ans[index[i]] <- ans[i]
                score[VS_index[index[i]]] = score[VS_index[index[i]]] + as.numeric(ans[i])
            }
            res = data.frame('SCORE' = score, 'VALUESYSTEMS' = names(score))
        ggplot(data = res, aes(x = VALUESYSTEMS, y = SCORE, 
                               fill = VALUESYSTEMS,
                               color = VALUESYSTEMS) )  + 
            geom_bar(stat = "identity") +
        theme_economist()+
        ylim(-10, 10)
        })
}

# Run the application 
shinyApp(ui = ui, server = server)
