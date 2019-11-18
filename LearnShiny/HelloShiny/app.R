library(shiny)
library(ggplot2)

myfun <- function(x){
  y <- x * 3 + 2 + rnorm(length(x))
  return(y)
}



ui <- fluidPage(
  sliderInput(inputId="num",
              label="Choose a number",
              value=10, min=2, max=100),
  plotOutput(outputId = 'myplot')
)



server <- function(input,output){
  output$myplot <- renderPlot({
  n <- input$num
  x= rnorm(n)
  da <- data.frame("X" = x, "Y" = myfun(x))
ggplot(data = da, mapping = aes(x = X,y = Y)) + geom_smooth(method = 'lm')
  })
}

shinyApp(server = server, ui = ui)
