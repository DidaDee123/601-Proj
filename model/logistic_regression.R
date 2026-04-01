# =============================================================================
# EXMD 601 - Team Project
# Logistic Regression Analysis: Cardiovascular Complications in Diabetic Patients
# Author: Christoforos
# =============================================================================
#
# Research Question: What factors predict cardiovascular complications in
#                    diabetic patients?
#
# This script fits logistic regression models as a benchmark to compare with
# the Random Forest model (David, Python). Two outcomes are modeled:
#   Outcome A: Cardiovascular complication within 10 years (stroke/HF/CKD)
#   Outcome B: ED visit within 1 year
#
# =============================================================================

# --- 0. Install / load required packages -------------------------------------

if (!require("pROC", quietly = TRUE)) {
  install.packages("pROC", repos = "https://cloud.r-project.org")
  library(pROC)
}

if (!require("ggplot2", quietly = TRUE)) {
  install.packages("ggplot2", repos = "https://cloud.r-project.org")
  library(ggplot2)
}

# --- 1. Load and inspect data ------------------------------------------------

# Path is relative to the script location (model/ directory)
data_path <- file.path("..", "..", "data", "cleaned_diabetic_patients_data.csv")
df <- read.csv(data_path, stringsAsFactors = FALSE)

cat("=== Dataset Dimensions ===\n")
cat("Rows:", nrow(df), " Columns:", ncol(df), "\n\n")

cat("=== Column Names ===\n")
print(names(df))
cat("\n")

cat("=== Structure ===\n")
str(df)
cat("\n")

cat("=== Summary Statistics ===\n")
summary(df)
cat("\n")

# Check outcome distribution (important for class imbalance)
cat("=== Outcome A Distribution (CV complication 10y) ===\n")
print(table(df$outcome_a_cv_complication_10y))
cat("Proportion positive:",
    mean(df$outcome_a_cv_complication_10y, na.rm = TRUE), "\n\n")

cat("=== Outcome B Distribution (ED visit 1y) ===\n")
print(table(df$outcome_b_ed_visit_1y))
cat("Proportion positive:",
    mean(df$outcome_b_ed_visit_1y, na.rm = TRUE), "\n\n")

# Check for remaining missing values
cat("=== Missing Values per Column ===\n")
print(colSums(is.na(df)))
cat("\n")

# --- 2. Define predictors ----------------------------------------------------

# All predictor columns (everything except the two outcome columns)
predictor_cols <- c(
  "age", "gender",
  "hba1c", "cholesterol", "glucose", "triglycerides", "creatinine",
  "has_hba1c", "has_cholesterol", "has_glucose",
  "has_triglycerides", "has_creatinine",
  "encounter_type_inpatient",
  "had_mi_20y"
)

# Verify all columns exist in the data
missing_cols <- setdiff(predictor_cols, names(df))
if (length(missing_cols) > 0) {
  warning("The following expected columns are missing from the data: ",
          paste(missing_cols, collapse = ", "))
}

# Build formula strings
formula_a <- as.formula(
  paste("outcome_a_cv_complication_10y ~", paste(predictor_cols, collapse = " + "))
)
formula_b <- as.formula(
  paste("outcome_b_ed_visit_1y ~", paste(predictor_cols, collapse = " + "))
)

cat("=== Model Formula (Outcome A) ===\n")
print(formula_a)
cat("\n")

# =============================================================================
# --- 3. Logistic Regression - Outcome A (CV Complication within 10 years) ----
# =============================================================================

cat("###################################################################\n")
cat("# OUTCOME A: Cardiovascular Complication within 10 Years          #\n")
cat("###################################################################\n\n")

model_a <- glm(formula_a, data = df, family = binomial(link = "logit"))

# --- 3a. Model summary -------------------------------------------------------
cat("=== Model Summary ===\n")
print(summary(model_a))
cat("\n")

cat("=== AIC ===\n")
cat("AIC:", AIC(model_a), "\n\n")

# --- 3b. Odds ratios and confidence intervals ---------------------------------
# Odds ratios = exp(coefficients); values > 1 indicate increased odds
coef_a <- summary(model_a)$coefficients
or_a   <- exp(coef(model_a))
ci_a   <- exp(confint.default(model_a))  # Wald-based confidence intervals

odds_table_a <- data.frame(
  Estimate   = coef_a[, "Estimate"],
  Std.Error  = coef_a[, "Std. Error"],
  z.value    = coef_a[, "z value"],
  p.value    = coef_a[, "Pr(>|z|)"],
  Odds.Ratio = or_a,
  OR_CI_Low  = ci_a[, 1],
  OR_CI_High = ci_a[, 2]
)

cat("=== Odds Ratios and 95% Confidence Intervals ===\n")
print(round(odds_table_a, 4))
cat("\n")

# Flag significant predictors (p < 0.05)
sig_a <- odds_table_a[odds_table_a$p.value < 0.05 &
                       rownames(odds_table_a) != "(Intercept)", ]
cat("=== Significant Predictors (p < 0.05) ===\n")
if (nrow(sig_a) > 0) {
  print(round(sig_a, 4))
} else {
  cat("No predictors reached significance at alpha = 0.05\n")
}
cat("\n")

# --- 3c. Model evaluation ----------------------------------------------------

# Predicted probabilities
pred_prob_a <- predict(model_a, type = "response")

# Confusion matrix at 0.5 threshold
pred_class_a <- ifelse(pred_prob_a >= 0.5, 1, 0)
actual_a     <- df$outcome_a_cv_complication_10y

cm_a <- table(Predicted = pred_class_a, Actual = actual_a)
cat("=== Confusion Matrix (threshold = 0.5) ===\n")
print(cm_a)
cat("\n")

# Extract TP, TN, FP, FN
TP_a <- ifelse("1" %in% rownames(cm_a) & "1" %in% colnames(cm_a), cm_a["1", "1"], 0)
TN_a <- ifelse("0" %in% rownames(cm_a) & "0" %in% colnames(cm_a), cm_a["0", "0"], 0)
FP_a <- ifelse("1" %in% rownames(cm_a) & "0" %in% colnames(cm_a), cm_a["1", "0"], 0)
FN_a <- ifelse("0" %in% rownames(cm_a) & "1" %in% colnames(cm_a), cm_a["0", "1"], 0)

accuracy_a    <- (TP_a + TN_a) / length(actual_a)
sensitivity_a <- ifelse((TP_a + FN_a) > 0, TP_a / (TP_a + FN_a), NA)
specificity_a <- ifelse((TN_a + FP_a) > 0, TN_a / (TN_a + FP_a), NA)
precision_a   <- ifelse((TP_a + FP_a) > 0, TP_a / (TP_a + FP_a), NA)
f1_a          <- ifelse(!is.na(precision_a) & !is.na(sensitivity_a) &
                          (precision_a + sensitivity_a) > 0,
                        2 * precision_a * sensitivity_a /
                          (precision_a + sensitivity_a), NA)

cat("=== Classification Metrics (Outcome A) ===\n")
cat("Accuracy:   ", round(accuracy_a, 4), "\n")
cat("Sensitivity:", round(sensitivity_a, 4), "\n")
cat("Specificity:", round(specificity_a, 4), "\n")
cat("Precision:  ", round(precision_a, 4), "\n")
cat("F1 Score:   ", round(f1_a, 4), "\n\n")

# ROC curve and AUC
roc_a <- roc(actual_a, pred_prob_a, quiet = TRUE)
auc_a <- auc(roc_a)
cat("=== AUC (Outcome A) ===\n")
cat("AUC:", as.numeric(auc_a), "\n\n")

# =============================================================================
# --- 4. Logistic Regression - Outcome B (ED Visit within 1 Year) -------------
# =============================================================================

cat("###################################################################\n")
cat("# OUTCOME B: ED Visit within 1 Year                              #\n")
cat("###################################################################\n\n")

model_b <- glm(formula_b, data = df, family = binomial(link = "logit"))

# --- 4a. Model summary -------------------------------------------------------
cat("=== Model Summary ===\n")
print(summary(model_b))
cat("\n")

cat("=== AIC ===\n")
cat("AIC:", AIC(model_b), "\n\n")

# --- 4b. Odds ratios and confidence intervals ---------------------------------
coef_b <- summary(model_b)$coefficients
or_b   <- exp(coef(model_b))
ci_b   <- exp(confint.default(model_b))

odds_table_b <- data.frame(
  Estimate   = coef_b[, "Estimate"],
  Std.Error  = coef_b[, "Std. Error"],
  z.value    = coef_b[, "z value"],
  p.value    = coef_b[, "Pr(>|z|)"],
  Odds.Ratio = or_b,
  OR_CI_Low  = ci_b[, 1],
  OR_CI_High = ci_b[, 2]
)

cat("=== Odds Ratios and 95% Confidence Intervals ===\n")
print(round(odds_table_b, 4))
cat("\n")

sig_b <- odds_table_b[odds_table_b$p.value < 0.05 &
                       rownames(odds_table_b) != "(Intercept)", ]
cat("=== Significant Predictors (p < 0.05) ===\n")
if (nrow(sig_b) > 0) {
  print(round(sig_b, 4))
} else {
  cat("No predictors reached significance at alpha = 0.05\n")
}
cat("\n")

# --- 4c. Model evaluation ----------------------------------------------------

pred_prob_b  <- predict(model_b, type = "response")
pred_class_b <- ifelse(pred_prob_b >= 0.5, 1, 0)
actual_b     <- df$outcome_b_ed_visit_1y

cm_b <- table(Predicted = pred_class_b, Actual = actual_b)
cat("=== Confusion Matrix (threshold = 0.5) ===\n")
print(cm_b)
cat("\n")

TP_b <- ifelse("1" %in% rownames(cm_b) & "1" %in% colnames(cm_b), cm_b["1", "1"], 0)
TN_b <- ifelse("0" %in% rownames(cm_b) & "0" %in% colnames(cm_b), cm_b["0", "0"], 0)
FP_b <- ifelse("1" %in% rownames(cm_b) & "0" %in% colnames(cm_b), cm_b["1", "0"], 0)
FN_b <- ifelse("0" %in% rownames(cm_b) & "1" %in% colnames(cm_b), cm_b["0", "1"], 0)

accuracy_b    <- (TP_b + TN_b) / length(actual_b)
sensitivity_b <- ifelse((TP_b + FN_b) > 0, TP_b / (TP_b + FN_b), NA)
specificity_b <- ifelse((TN_b + FP_b) > 0, TN_b / (TN_b + FP_b), NA)
precision_b   <- ifelse((TP_b + FP_b) > 0, TP_b / (TP_b + FP_b), NA)
f1_b          <- ifelse(!is.na(precision_b) & !is.na(sensitivity_b) &
                          (precision_b + sensitivity_b) > 0,
                        2 * precision_b * sensitivity_b /
                          (precision_b + sensitivity_b), NA)

cat("=== Classification Metrics (Outcome B) ===\n")
cat("Accuracy:   ", round(accuracy_b, 4), "\n")
cat("Sensitivity:", round(sensitivity_b, 4), "\n")
cat("Specificity:", round(specificity_b, 4), "\n")
cat("Precision:  ", round(precision_b, 4), "\n")
cat("F1 Score:   ", round(f1_b, 4), "\n\n")

roc_b <- roc(actual_b, pred_prob_b, quiet = TRUE)
auc_b <- auc(roc_b)
cat("=== AUC (Outcome B) ===\n")
cat("AUC:", as.numeric(auc_b), "\n\n")

# =============================================================================
# --- 5. Comparison Table ------------------------------------------------------
# =============================================================================

cat("###################################################################\n")
cat("# MODEL COMPARISON SUMMARY                                       #\n")
cat("###################################################################\n\n")

comparison <- data.frame(
  Metric      = c("AIC", "AUC", "Accuracy", "Sensitivity", "Specificity",
                   "Precision", "F1 Score"),
  Outcome_A   = c(AIC(model_a), as.numeric(auc_a),
                   accuracy_a, sensitivity_a, specificity_a,
                   precision_a, f1_a),
  Outcome_B   = c(AIC(model_b), as.numeric(auc_b),
                   accuracy_b, sensitivity_b, specificity_b,
                   precision_b, f1_b)
)

cat("=== Side-by-Side Comparison ===\n")
print(comparison, digits = 4, row.names = FALSE)
cat("\n")

# =============================================================================
# --- 6. Visualizations --------------------------------------------------------
# =============================================================================

# --- 6a. Odds Ratio Forest Plot (Outcome A) ----------------------------------
# This plot shows the odds ratio and 95% CI for each predictor.
# The dashed line at OR=1 means "no effect"; points to the right indicate
# increased odds of cardiovascular complication.

# Prepare data (exclude intercept)
or_plot_data <- odds_table_a[-1, ]  # drop intercept row
or_plot_data$Variable <- rownames(or_plot_data)

# Reorder by odds ratio for readability
or_plot_data$Variable <- factor(or_plot_data$Variable,
                                levels = or_plot_data$Variable[
                                  order(or_plot_data$Odds.Ratio)])

p_forest <- ggplot(or_plot_data, aes(x = Odds.Ratio, y = Variable)) +
  geom_point(size = 3) +
  geom_errorbarh(aes(xmin = OR_CI_Low, xmax = OR_CI_High), height = 0.2) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "red") +
  labs(
    title = "Outcome A: Odds Ratios with 95% CI",
    subtitle = "Cardiovascular complication within 10 years",
    x = "Odds Ratio (log scale)",
    y = ""
  ) +
  scale_x_log10() +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"))

# Save to file
ggsave("outcome_a_odds_ratios.png", plot = p_forest,
       width = 8, height = 6, dpi = 150)
cat("Saved: outcome_a_odds_ratios.png\n")

# --- 6b. Odds Ratio Forest Plot (Outcome B) ----------------------------------

or_plot_data_b <- odds_table_b[-1, ]
or_plot_data_b$Variable <- rownames(or_plot_data_b)
or_plot_data_b$Variable <- factor(or_plot_data_b$Variable,
                                  levels = or_plot_data_b$Variable[
                                    order(or_plot_data_b$Odds.Ratio)])

p_forest_b <- ggplot(or_plot_data_b, aes(x = Odds.Ratio, y = Variable)) +
  geom_point(size = 3) +
  geom_errorbarh(aes(xmin = OR_CI_Low, xmax = OR_CI_High), height = 0.2) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "red") +
  labs(
    title = "Outcome B: Odds Ratios with 95% CI",
    subtitle = "ED visit within 1 year",
    x = "Odds Ratio (log scale)",
    y = ""
  ) +
  scale_x_log10() +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"))

ggsave("outcome_b_odds_ratios.png", plot = p_forest_b,
       width = 8, height = 6, dpi = 150)
cat("Saved: outcome_b_odds_ratios.png\n")

# --- 6c. ROC Curves for Both Outcomes ----------------------------------------

# Build a combined data frame for ggplot
roc_df_a <- data.frame(
  FPR = 1 - roc_a$specificities,
  TPR = roc_a$sensitivities,
  Model = paste0("Outcome A (AUC = ", round(as.numeric(auc_a), 3), ")")
)
roc_df_b <- data.frame(
  FPR = 1 - roc_b$specificities,
  TPR = roc_b$sensitivities,
  Model = paste0("Outcome B (AUC = ", round(as.numeric(auc_b), 3), ")")
)
roc_df <- rbind(roc_df_a, roc_df_b)

p_roc <- ggplot(roc_df, aes(x = FPR, y = TPR, color = Model)) +
  geom_line(linewidth = 1) +
  geom_abline(linetype = "dashed", color = "grey50") +
  labs(
    title = "ROC Curves: Logistic Regression",
    x = "False Positive Rate (1 - Specificity)",
    y = "True Positive Rate (Sensitivity)",
    color = NULL
  ) +
  coord_equal() +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold"),
    legend.position = "bottom"
  )

ggsave("roc_curves_both_outcomes.png", plot = p_roc,
       width = 7, height = 7, dpi = 150)
cat("Saved: roc_curves_both_outcomes.png\n")

cat("\n=== Analysis complete. ===\n")
