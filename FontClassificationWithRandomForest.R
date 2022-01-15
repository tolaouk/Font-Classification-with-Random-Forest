rm(list = ls())
# ===========================================================================
library(dplyr)
library(caret)
library(ggplot2)
library(tree)
library(randomForest)
# ===========================================================================
# Problem 0
file_wd = "/Users/jerrychien/Desktop/OneDrive - University Of Houston/6350 - Statistical Learning and Data Mining/HW/HW 3/fonts"
selected_file = c("GILL.csv", "LEELAWADEE.csv", "ROMAN.csv", "TECHNIC.csv") # Each class around 1300. Report now base on this case.
file_class = data.frame()
for (i in c(1:4)){
  file_class[paste("Class", i), "File"] = selected_file[i]
}

# Read the 4 csv files and assign to 4 different data frame and delete the unnecessary columns. 
# Filter all the 4 data frames with strength = 0.4 and italic = 0 and assign them to 4 different class. 
col_to_be_skipped = c("fontVariant", "m_label", "orientation", "m_top", "m_left", "originalH", "originalW", "h", "w")
for (i in 1:length(selected_file)){
  assign(paste("CL", i, sep = ""), 
         filter(select(read.csv(paste(file_wd, "/", selected_file[i], sep = ""), skipNul = T), -col_to_be_skipped), strength == 0.4 & italic == 0)[, -c(1, 2, 3)])
}
CL_list = c("CL1", "CL2", "CL3", "CL4")

# Print out the size of each class and the total size of 4 classes.
class_size = data.frame()
N = 0
for (i in CL_list){
  class_size[i, "Size"] = dim(get(i))[1]
  N = N + dim(get(i))[1]
}
class_size["Total", "Size"] = N
class_size[, "Percentage"] = round((class_size[, "Size"] / N) * 100)

# Combine all the 4 classes and assign to a data frame.
DATA = rbind(CL1, CL2, CL3, CL4)

# True Class for each case
TRUC = data.frame()
for (i in CL_list){
  temp = as.data.frame(rep(substring(i, 3), dim(get(i))[1]))
  colnames(temp) = "TRUC"
  TRUC = rbind(TRUC, temp)
}

# PCA
PCA = prcomp(DATA, center = TRUE, scale = TRUE) # Use built-in PCA
L = PCA$sdev^2 # Eigenvalue for finding the PEV >= 90%

PEV = NULL
for (m in 1:400){
  PEV[m] = sum(L[c(1:m)]) / 400 * 100
}
smallest_r = sum(PEV < 90) + 1 # Smallest r to have PEV >= 90%

ZDATA = PCA$x[ ,c(1:smallest_r)] # Truncate the data to first "r" principal components

final_ZDATA = cbind(TRUC, ZDATA) # Final data after PCA with true class.

# ===========================================================================
# Problem 1
## 1.2
data_split = function(portion = 0.2, nclass = 4, df){
  test_set = data.frame()
  training_set = data.frame()
  size_test_training = data.frame()
  for (i in 1:nclass){
    temp = filter(df, TRUC == i)
    sampled_number = sample(dim(temp)[1], dim(temp)[1] * portion)
    
    temp_training = temp[-sampled_number, ]
    temp_test = temp[sampled_number, ]
    
    test_set = rbind(test_set, temp_test)
    training_set = rbind(training_set, temp_training)
    
    size_test_training[paste("CL", i, sep = ""), "Test"] = dim(temp_test)[1]
    size_test_training[paste("CL", i, sep = ""), "Training"] = dim(temp_training)[1]
  }
  size_test_training["Total", "Test"] = dim(test_set)[1]
  size_test_training["Total", "Training"] = dim(training_set)[1]
  
  return(list(training_set, test_set, size_test_training))
}
set.seed(20211004) # Use seed = 20211004 to get same data set as HW2
split_data = data_split(portion = 0.2, nclass = 4, df = final_ZDATA) 
training_set = split_data[[1]]
test_set = split_data[[2]]
size_test_training = split_data[[3]]

# ===========================================================================
# Problem 3
## 3.1 Built the random forest with 100 trees.
rf_100 = randomForest(x = training_set[, -1], y = as.factor(training_set[, 1]), ntree = 100, importance = TRUE)
pred_training_100 = predict(rf_100, training_set[, -1])
pred_test_100 = predict(rf_100, test_set[, -1])

## 3.2 Calculate the global performance and confusion matrix
global_acc = data.frame("Set" = c("Training", "Test"), 
                        "Accuracy" = c(round(mean(pred_training_100 == training_set[, 1]) * 100, 1), round(mean(pred_test_100 == test_set[, 1]) * 100, 1)))

conf_training_100 = round(prop.table(confusionMatrix(data = pred_training_100, reference = as.factor(training_set[, 1]))$table, margin = 2) * 100, 1)
conf_test_100 = round(prop.table(confusionMatrix(data = pred_test_100, reference = as.factor(test_set[, 1]))$table, margin = 2) * 100, 1)

# ===========================================================================
# Problem 4
## 4.1 Built the random forest with 200 and 300 trees.
rf_200 = randomForest(x = training_set[, -1], y = as.factor(training_set[, 1]), ntree = 200, importance = TRUE)
pred_training_200 = predict(rf_200, training_set[, -1])
pred_test_200 = predict(rf_200, test_set[, -1])
conf_test_200 = round(prop.table(confusionMatrix(data = pred_test_200, reference = as.factor(test_set[, 1]))$table, margin = 2) * 100, 1)

rf_300 = randomForest(x = training_set[, -1], y = as.factor(training_set[, 1]), ntree = 300, importance = TRUE)
pred_training_300 = predict(rf_300, training_set[, -1])
pred_test_300 = predict(rf_300, test_set[, -1])
conf_test_300 = round(prop.table(confusionMatrix(data = pred_test_300, reference = as.factor(test_set[, 1]))$table, margin = 2) * 100, 1)

## 4.2 Compare the performance and select the best tree number.
diff_number_tree_comp = data.frame()
for (i in 1:3){
  diff_number_tree_comp[i, "Number_of_Trees"] = i * 100
  temp_test_acc = mean(get(paste("pred_test_", i * 100, sep = "")) == test_set[, 1])
  diff_number_tree_comp[i, "Test_Accuracy"] = round(temp_test_acc * 100, 1)
  SE = sqrt(temp_test_acc * (1 - temp_test_acc) / dim(test_set)[1])
  diff_number_tree_comp[i, "LL_90_Test"] = round((temp_test_acc - 1.6 * SE) * 100, 1)
  diff_number_tree_comp[i, "UL_90_Test"] = round((temp_test_acc + 1.6 * SE) * 100, 1)
}

ggplot(data = diff_number_tree_comp) +
  geom_ribbon(aes(x = Number_of_Trees, ymax = UL_90_Test, ymin = LL_90_Test, fill = "90% C.I."), alpha = 0.5) +
  geom_point(aes(x = Number_of_Trees, y = Test_Accuracy)) +
  geom_line(aes(x = Number_of_Trees, y = Test_Accuracy, color = "Test")) +
  geom_text(aes(Number_of_Trees, Test_Accuracy, label = Test_Accuracy), hjust = 0.5, vjust = 1.5, size = 8) +
  theme(text = element_text(size = 25)) +
  ylab("Accuracy (%)") + xlab("Tree Number") + expand_limits(x = c(90, 310), y = c(80, 100)) +
  scale_x_discrete(limits = c(100, 200, 300)) +
  scale_color_manual(name = "Accuracy", values = ("Test" = "black")) + 
  scale_fill_manual(name = "C.I.", values = c("90% C.I." = "green")) +
  theme(legend.position = c(0.85, 0.90), legend.text = element_text(size = 20), legend.title = element_text(size = 20), legend.box = "horizontal")
best_tree_number = as.integer(diff_number_tree_comp[diff_number_tree_comp["Test_Accuracy"] == max(diff_number_tree_comp["Test_Accuracy"])][1])

# ===========================================================================
## 5.1 Feature Importance
sorted_IM = sort(get(paste("rf_", best_tree_number, sep = ""))$importance[, "MeanDecreaseAccuracy"], decreasing = T)[1:10]
feature_IM = data.frame()
for (i in 1:10) {
  feature_number = as.integer(substring(names(sorted_IM)[i], 3))
  feature_IM[i, "Feature"] = feature_number
  feature_IM[i, "Importance"] = round(sorted_IM[i] * 100, 1)
  feature_IM[i, "Eigenvalue"] = L[feature_number]
}

ggplot() +
  geom_point(aes(x = c(1:10), y = feature_IM[, "Importance"])) +
  geom_line(aes(x = c(1:10), y = feature_IM[, "Importance"])) +
  geom_text(aes(c(1:10) + c(-0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0), feature_IM[,"Importance"], label = paste("Z", feature_IM[, "Feature"], sep = "")), hjust = 0.5, vjust = 1.5, size = 8) +
  theme(text = element_text(size = 25)) +
  ylab("Importance (%)") + xlab("Importance Ranking") + expand_limits(x = c(1,10), y = c(0, 2.5)) +
  scale_x_discrete(limits = factor(c(1:10)))

## 5.3
ggplot() +
  geom_point(aes(x = feature_IM[, "Eigenvalue"], y = feature_IM[, "Importance"])) +
  geom_text(aes(feature_IM[, "Eigenvalue"] + c(0, 0, 0, 2.5, -2.5, 0, 0, 0, 0, 0), feature_IM[, "Importance"] + c(0, 0, 0, 0.5, 0.5, 0, 0, 0, 0, 0), label = paste("Z", feature_IM[, "Feature"], sep = "")), hjust = 0.5, vjust = 1.5, size = 5) +
  theme(text = element_text(size = 25)) +
  ylab("Accuracy Loss (%)") + xlab("Eigenvalue") + expand_limits(x = c(0,100), y = c(0, 2.5))

cor_L_IM = cor(feature_IM[, "Eigenvalue"], feature_IM[, "Importance"])

# ===========================================================================
# Problem 6
## 6.1
conf_test_best_tree = round(prop.table(get(paste("conf_test_", best_tree_number, sep = "")), margin = 2) * 100, 1)
best_tree_acc = round(mean(get(paste("pred_test_", best_tree_number, sep = "")) == test_set[, 1]) * 100, 1)

paired_CL = combn(c(1:length(selected_file)), 2)
misclassifed = data.frame()
for (i in c(1:dim(paired_CL)[2])){
  misclassifed[i, "CLi"] = paired_CL[1, i]
  misclassifed[i, "CLj"] = paired_CL[2, i]
  misclassifed[i, "Rate"] = conf_test_best_tree[paired_CL[1, i], paired_CL[2, i]] + conf_test_best_tree[paired_CL[2, i], paired_CL[1, i]]
}

## 6.2
for (i in c(1:dim(paired_CL)[2])){
  training_set_pair = filter(training_set, TRUC == paired_CL[1, i] | TRUC == paired_CL[2, i])
  test_set_pair = filter(test_set, TRUC == paired_CL[1, i] | TRUC == paired_CL[2, i])
  pred_test_pair = predict(randomForest(x = training_set_pair[, -1], y = as.factor(training_set_pair[, 1]), ntree = best_tree_number, importance = TRUE), test_set_pair[, -1])
  assign(paste("conf_test_pair_", paired_CL[1, i], "_", paired_CL[2, i], sep = ""), round(prop.table(confusionMatrix(as.factor(pred_test_pair), as.factor(test_set_pair[, 1]))$table, margin = 2) * 100, 1))
}

## 6.3.1 Enlarge the tree number
different_tree = data.frame()
for (i in seq(0, 500, 10)[-1]){
  temp_rf = randomForest(x = training_set[, -1], y = as.factor(training_set[, 1]), ntree = i, importance = TRUE)
  temp_predict = predict(temp_rf, test_set[, -1])
  different_tree[i/10, "Tree_Number"] = i
  different_tree[i/10, "Accuracy"] = round(mean(temp_predict == test_set[, 1]) * 100, 1)
}

ggplot(data = different_tree) +
  geom_line(aes(x = Tree_Number, y = Accuracy)) +
  theme(text = element_text(size = 25)) +
  ylab("Accuracy (%)") + xlab("Tree Number") + expand_limits(x = c(0, 500), y = c(80, 100)) +
  scale_x_discrete(limits = c(10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500)) +
  theme(plot.margin = unit(c(10, 15, 10, 10), "point"))

## 6.3.2 Change the number of features of be used in each tree in RF model
different_feature = data.frame()
for (i in 1:(smallest_r - 1)){
  temp_rf = randomForest(x = training_set[, -1], y = as.factor(training_set[, 1]), ntree = best_tree_number, mtry = i, importance = TRUE)
  temp_predict = predict(temp_rf, test_set[, -1])
  different_feature[i, "Feature_Number"] = i
  different_feature[i, "Accuracy"] = round(mean(temp_predict == test_set[, 1]) * 100, 1)
}

ggplot(data = different_feature) +
  geom_line(aes(x = Feature_Number, y = Accuracy)) +
  theme(text = element_text(size = 25)) +
  ylab("Accuracy (%)") + xlab("Feature Number") + expand_limits(x = c(0, 62), y = c(80, 100)) +
  scale_x_discrete(limits = c(1, 5, 7, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 61))

## 6.3.3 k-means clustering
training_set_k_means = kmeans(training_set[, -1], centers = 4, nstart = 20)
cent = training_set_k_means$centers
cluster_training_set = cbind(training_set_k_means$cluster, training_set)
colnames(cluster_training_set)[1] = "cluster"

purity = data.frame()
for (i in 1:length(selected_file)){
  temp = filter(as.data.frame(cluster_training_set), cluster == i)
  gini = sum(prop.table(table(temp$TRUC)) * (1 - prop.table(table(temp$TRUC))))
  purity[i, "Cluster"] = i
  purity[i, "Gini"] = round(gini, 2)
  assign(paste("rf_G", i, sep = ""), randomForest(x = temp[, -c(1,2)], y = as.factor(temp[, 2]), ntree = best_tree_number, importance = TRUE))
}

pred_test_k_means_rf = NULL
for (i in c(1:dim(test_set)[1])){
  distance = data.frame()
  for (j in 1:length(selected_file)){
    distance[j, "Cluster"] = j
    distance[j, "Distance"] = dist(rbind(test_set[i, -1], cent[j, ]))[1]
  }
  closest_G = distance[min(distance[, "Distance"]) == distance[, "Distance"], "Cluster"]
  pred_test_k_means_rf[i] = predict(get(paste("rf_G", closest_G, sep = "")), test_set[i, -1])
}

conf_test_k_means_rf = round(prop.table(confusionMatrix(as.factor(pred_test_k_means_rf), as.factor(test_set[, 1]))$table, margin = 2) * 100, 1)
k_means_rf_acc = round(mean(pred_test_k_means_rf == test_set[, 1]) * 100, 1)

## 6.3.4 6 RF classifiers for 6 pairs of classes
for (i in c(1:dim(paired_CL)[2])){
  training_set_CLi_CLj = filter(training_set, TRUC == paired_CL[1, i] | TRUC == paired_CL[2, i])
  assign(paste("rf_", paired_CL[1, i], paired_CL[2, i], sep = ""), randomForest(x = training_set_CLi_CLj[, -1], y = as.factor(training_set_CLi_CLj[, 1]), ntree = best_tree_number, importance = TRUE))
}

pred_test_paired_rf = NULL
for (i in c(1:dim(test_set)[1])){
  for (j in c(1:dim(paired_CL)[2])){
    assign(paste("pred_rf_", paired_CL[1, j], paired_CL[2, j], sep = ""), predict(get(paste("rf_", paired_CL[1, j], paired_CL[2, j], sep = "")), test_set[i, ]))
  }
  vote = data.frame("Class" = c(1:length(selected_file)))
  vote[1, "Vote"] = (pred_rf_12 == 1) + (pred_rf_13 == 1) + (pred_rf_14 == 1)
  vote[2, "Vote"] = (pred_rf_12 == 2) + (pred_rf_23 == 2) + (pred_rf_24 == 2)
  vote[3, "Vote"] = (pred_rf_13 == 3) + (pred_rf_23 == 3) + (pred_rf_34 == 3)
  vote[4, "Vote"] = (pred_rf_14 == 4) + (pred_rf_24 == 4) + (pred_rf_34 == 4)
  max_freq = vote[vote["Vote"] == max(vote["Vote"]), "Class"]
  if (length(max_freq) != 1){
    pred_test_paired_rf[i] = sample(max_freq, 1)
  } else{
    pred_test_paired_rf[i] = max_freq
  }
}

conf_test_paired_rf = round(prop.table(confusionMatrix(as.factor(pred_test_paired_rf), as.factor(test_set[, 1]))$table, margin = 2) * 100, 1)
paired_rf_acc = round(mean(pred_test_paired_rf == test_set[, 1]) * 100, 1)

## 6.3.5 Bag of RF classifiers
bag_rf = 5
for (i in (1:bag_rf)){
  set.seed(i)
  assign(paste("sub_training_set_", i, sep = ""), data_split(portion = 0.2, nclass = 4, df = training_set)[[1]])
  assign(paste("rf_tr", i, sep = ""), randomForest(x = get(paste("sub_training_set_", i, sep = ""))[, -1], y = as.factor(get(paste("sub_training_set_", i, sep = ""))[, 1]), ntree = best_tree_number, importance = TRUE))
}

pred_test_bag_rf = NULL
for (i in c(1:dim(test_set)[1])){
  for (j in c(1:bag_rf)){
    assign(paste("pred_rf_tr", j, sep = ""), predict(get(paste("rf_tr", j, sep = "")), test_set[i, ]))
  }
  vote2 = data.frame("Class" = c(1:length(selected_file)))
  vote2[1, "Vote"] = (pred_rf_tr1 == 1) + (pred_rf_tr2 == 1) + (pred_rf_tr3 == 1) + (pred_rf_tr4 == 1) + (pred_rf_tr5 == 1)
  vote2[2, "Vote"] = (pred_rf_tr1 == 2) + (pred_rf_tr2 == 2) + (pred_rf_tr3 == 2) + (pred_rf_tr4 == 2) + (pred_rf_tr5 == 2)
  vote2[3, "Vote"] = (pred_rf_tr1 == 3) + (pred_rf_tr2 == 3) + (pred_rf_tr3 == 3) + (pred_rf_tr4 == 3) + (pred_rf_tr5 == 3)
  vote2[4, "Vote"] = (pred_rf_tr1 == 4) + (pred_rf_tr2 == 4) + (pred_rf_tr3 == 4) + (pred_rf_tr4 == 4) + (pred_rf_tr5 == 4)
  max_freq2 = vote2[vote2["Vote"] == max(vote2["Vote"]), "Class"]
  if (length(max_freq2) != 1){
    pred_test_bag_rf[i] = sample(max_freq2, 1)
  } else{
    pred_test_bag_rf[i] = max_freq2
  }
}

conf_test_bag_rf = round(prop.table(confusionMatrix(as.factor(pred_test_bag_rf), as.factor(test_set[, 1]))$table, margin = 2) * 100, 1)
bag_rf_acc = round(mean(pred_test_bag_rf == test_set[, 1]) * 100, 1)

acc_comp = data.frame("Case Study" = c(paste("Best Tree Number =", best_tree_number), "K-Means Clustering", "Paired Classes", "Bag of RF"),
                      "Accuracy" = c(best_tree_acc, k_means_rf_acc, paired_rf_acc, bag_rf_acc))
