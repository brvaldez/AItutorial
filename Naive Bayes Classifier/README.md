# Naive Bayes Classifier — Breast Cancer Diagnosis

A machine learning project that uses Gaussian Naive Bayes to classify breast tumors as benign or malignant based on numeric cell measurements from medical imaging.

## Dataset

- Breast Cancer Wisconsin 
- 30 continuous features (e.g., mean radius, mean texture, mean area)
- 2 classes: Malignant (0) = cancer, Benign (1) = no cancer
- No missing values; no preprocessing required

Why is Gaussian Naive Bayes appropriate for this dataset?

Gaussian Naive Bayes is a great choice here because our data is made up of continuous numbers (numerical variables) 
(like measurements for radius, area, and texture) rather than categories (categorical varibles) like "red" or "blue".
Because the features in this dataset are continuous biological measurements, they naturally follow a normal
(Gaussian) distribution, where values cluster around a central mean. Gaussian Naive Bayes is ideal for this 
scenario because it explicitly assumes this distribution to calculate class probabilities.

How the model makes predictions?

Learning: First, it looks at the training data and calculates the "average" look of a Benign tumor and the "average" look of a Malignant tumor for every feature.
Testing: When we give it a new tumor to classify, it calculates the probability (likelihood) that the new measurements belong to the Benign group versus the Malignant group. It does this by checking where the new numbers fall on the Bell Curves it learned earlier.
Decision: It combines the probabilities from all 30 features. Whichever group has the higher total probability score wins, and that becomes the prediction.

What different types of classification errors could mean in a medical diagnosis setting?

False Negative: This happens when the model predicts a tumor is Benign (safe), but it is actually Malignant (cancer). This is the worst-case scenario because a sick patient might be sent home without treatment, allowing the cancer to grow.

False Positive: This happens when the model predicts Malignant, but the tumor is actually Benign. This isn't deadly, but it causes a lot of unnecessary stress for the patient and leads to expensive, painful procedures (like biopsies) that weren't actually needed.