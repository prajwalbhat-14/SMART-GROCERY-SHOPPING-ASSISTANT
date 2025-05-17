# Load required libraries
install.packages("shiny")
install.packages("dplyr")
install.packages("ggplot2")
install.packages("readr")
install.packages("arules")
install.packages("caret")
install.packages("randomForest")  # Ensure rf method works

library(shiny)
library(dplyr)
library(ggplot2)
library(readr)
library(arules)
library(caret)
library(randomForest)

# Load cleaned grocery dataset
df <- read_csv("C:/Users/Asus/Downloads/grocery_dataset_updated.csv", show_col_types = FALSE)

# Add Time column if not present (for Apriori grouping)
if (!"Time" %in% colnames(df)) {
  df$Time <- rep(1:ceiling(nrow(df) / 5), each = 5, length.out = nrow(df))
}

# -------- Train ML Model outside Shiny --------
target_column <- "Stock_Available"  # <-- Replace with actual column name for prediction

# Ensure target column exists
if (!target_column %in% colnames(df)) {
  stop("Target column does not exist in the dataset.")
}

# Remove rows with NA target and convert to factor
df <- df[!is.na(df[[target_column]]), ]
df[[target_column]] <- as.factor(df[[target_column]])

# Check for multiple classes
if (length(unique(df[[target_column]])) < 2) {
  stop("Target column must have at least two classes for classification.")
}

# Optionally exclude non-feature columns
exclude_columns <- c("Item_names", "Brand", "Time")
features <- setdiff(colnames(df), c(target_column, exclude_columns))
df_model <- df[, c(features, target_column)]

# Split data
set.seed(123)
trainIndex <- createDataPartition(df_model[[target_column]], p = 0.8, list = FALSE)
trainData <- df_model[trainIndex, ]
testData <- df_model[-trainIndex, ]

# Train model
model <- train(
  as.formula(paste(target_column, "~ .")),
  data = trainData,
  method = "rf",
  trControl = trainControl(method = "cv", number = 5)
)

# Evaluate
predictions <- predict(model, newdata = testData)
confusion_matrix <- confusionMatrix(predictions, as.factor(testData[[target_column]]))
print(confusion_matrix)

# ----------- Shiny UI ------------
ui <- fluidPage(
  titlePanel("🛒 Smart Grocery Shopping Assistant"),
  
  sidebarLayout(
    sidebarPanel(
      selectInput("item", "Select Grocery Item:", choices = unique(df$Item_names)),
      uiOutput("brand_ui"),
      numericInput("quantity", "Enter Quantity:", value = 1, min = 1),
      actionButton("add", "Add to Cart"),
      actionButton("clear", "Clear Cart"),
      actionButton("order", "Order Now", class = "btn-success"),
      hr(),
      selectInput("remove_item", "Remove Item:", choices = NULL),
      actionButton("remove", "Remove from Cart", class = "btn-danger")
    ),
    
    mainPanel(
      h3("Cart Summary"),
      tableOutput("cart"),
      textOutput("total_cost"),
      h4("📊 Cart Metrics"),
      tableOutput("metrics"),
      plotOutput("price_chart"),
      h4(textOutput("order_status"), style = "color: green; font-weight: bold;"),
      hr(),
      h4("🧠 Recommended Combos (Apriori)"),
      tableOutput("recommendations")
    )
  )
)

# ------------ Shiny Server ------------
server <- function(input, output, session) {
  
  # Brand UI
  output$brand_ui <- renderUI({
    req(input$item)
    brand_data <- df %>% filter(Item_names == input$item) %>% distinct(Brand, Price)
    if (nrow(brand_data) == 1) return(NULL)
    choices <- brand_data %>%
      mutate(label = paste0(Brand, " - ₹", Price)) %>%
      { setNames(.$Brand, .$label) }
    selectInput("brand", "Select Brand & Price:", choices = choices)
  })
  
  # Reactive cart
  cart <- reactiveVal(data.frame(Item = character(), Brand = character(), Quantity = numeric(), Price = numeric(), Cost = numeric(), stringsAsFactors = FALSE))
  
  # Add item
  observeEvent(input$add, {
    req(input$item, input$quantity)
    selected_brand <- if (!is.null(input$brand)) input$brand else {
      df %>% filter(Item_names == input$item) %>% pull(Brand) %>% .[1]
    }
    price <- df %>% filter(Item_names == input$item, Brand == selected_brand) %>% pull(Price) %>% .[1]
    cart_data <- cart()
    existing <- which(cart_data$Item == input$item & cart_data$Brand == selected_brand)
    
    if (length(existing)) {
      cart_data[existing, "Quantity"] <- cart_data[existing, "Quantity"] + input$quantity
      cart_data[existing, "Cost"] <- cart_data[existing, "Quantity"] * cart_data[existing, "Price"]
    } else {
      new_entry <- data.frame(Item = input$item, Brand = selected_brand, Quantity = input$quantity, Price = price, Cost = input$quantity * price, stringsAsFactors = FALSE)
      cart_data <- rbind(cart_data, new_entry)
    }
    
    cart(cart_data)
    updateSelectInput(session, "remove_item", choices = paste(cart_data$Item, "-", cart_data$Brand))
  })
  
  # Remove item
  observeEvent(input$remove, {
    req(input$remove_item)
    cart_data <- cart()
    split <- strsplit(input$remove_item, " - ")[[1]]
    cart_data <- cart_data[!(cart_data$Item == split[1] & cart_data$Brand == split[2]), ]
    cart(cart_data)
    updateSelectInput(session, "remove_item", choices = if (nrow(cart_data) > 0) paste(cart_data$Item, "-", cart_data$Brand) else NULL)
  })
  
  # Clear cart
  observeEvent(input$clear, {
    cart(data.frame(Item = character(), Brand = character(), Quantity = numeric(), Price = numeric(), Cost = numeric(), stringsAsFactors = FALSE))
    updateSelectInput(session, "remove_item", choices = NULL)
    output$order_status <- renderText("")
  })
  
  # Order
  observeEvent(input$order, {
    cart_data <- cart()
    if (nrow(cart_data) == 0) {
      output$order_status <- renderText("❌ Cart is empty! Add items before ordering.")
    } else {
      total <- sum(cart_data$Cost)
      output$order_status <- renderText(paste("✅ Order placed! Total: ₹", round(total, 2)))
    }
  })
  
  # Render outputs
  output$cart <- renderTable(cart())
  
  output$total_cost <- renderText({
    paste("Total Cost: ₹", round(sum(cart()$Cost, na.rm = TRUE), 2))
  })
  
  output$metrics <- renderTable({
    cart_data <- cart()
    if (nrow(cart_data) == 0) return(NULL)
    most_frequent_item <- cart_data %>%
      group_by(Item) %>%
      summarise(TotalQty = sum(Quantity)) %>%
      arrange(desc(TotalQty)) %>%
      slice(1) %>%
      pull(Item)
    
    data.frame(
      `Total Unique Items` = nrow(cart_data),
      `Total Quantity` = sum(cart_data$Quantity),
      `Average Cost/Item` = round(mean(cart_data$Cost), 2),
      `Unique Brands` = length(unique(cart_data$Brand)),
      `Most Frequent Item` = most_frequent_item,
      check.names = FALSE
    )
  })
  
  output$price_chart <- renderPlot({
    cart_data <- cart()
    if (nrow(cart_data) > 0) {
      ggplot(cart_data, aes(x = Item, y = Cost, fill = Item)) +
        geom_bar(stat = "identity") +
        theme_minimal() +
        labs(title = "Spending by Item", y = "Cost (₹)", x = "Item") +
        theme(axis.text.x = element_text(angle = 45, hjust = 1))
    }
  })
  
  output$recommendations <- renderTable({
    if (nrow(df) >= 10) {
      transactions <- as(split(df$Item_names, df$Time), "transactions")
      rules <- apriori(transactions, parameter = list(supp = 0.01, conf = 0.1, minlen = 2))
      top_rules <- head(sort(rules, by = "lift"), 5)
      
      rule_df <- as(top_rules, "data.frame") %>%
        mutate(
          LHS = labels(lhs(top_rules)),
          RHS = labels(rhs(top_rules))
        ) %>%
        select(LHS, RHS, support, confidence, lift) %>%
        rename(
          `LHS (If)` = LHS,
          `RHS (Then)` = RHS,
          `Support` = support,
          `Confidence` = confidence,
          `Lift` = lift
        )
      
      rule_df
    } else {
      data.frame(Message = "Not enough data for Apriori recommendations.")
    }
  })
}

# Run the app
shinyApp(ui, server)
