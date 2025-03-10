import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import joblib
import math
import seaborn as sns
import matplotlib.pyplot as plt

# Load the prepared EEG data from the CSV file
prepared_data = pd.read_csv('prepared_eeg_data13_1.csv')

# Remove class 3 (Label "None") due to insufficient instances
prepared_data = prepared_data[prepared_data['Label_Encoded'] != 3]

# Split the dataset into features (X) and labels (y)
X = prepared_data[['Duration', 'Max_Amplitude', 'Min_Amplitude', 'Mean_Amplitude', 'Std_Amplitude', 'Spike_Count', 'Spike_Rate', 'Change_in_Spike_Rate']]
y = prepared_data['Label_Encoded']

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Calculate the test size dynamically as a percentage of the total dataset size
test_size_percentage = 0.2# You can adjust this percentage as needed
test_size = int(len(prepared_data) * test_size_percentage)
print("\nNo of samples used for testing", test_size, "out of", int(len(prepared_data)))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

class_counts = y_train.value_counts()

minority_class = class_counts.idxmin()  # Get the class with the minimum count
minority_class_samples = class_counts.min()  # Number of samples in the minority class
print("\nMinority class is", minority_class, "and the no of minority samples =", minority_class_samples)

# Calculate k_neighbors dynamically
k_neighbors = math.isqrt(minority_class_samples)

# Apply SMOTE only to the minority class (class 2) with dynamically calculated k_neighbors
smote = SMOTE(sampling_strategy={2: int(0.5 * len(y_train))}, k_neighbors=k_neighbors, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Initialize the SVM classifier
svm_classifier = SVC(kernel='rbf', probability=True, random_state=42)

# Perform hyperparameter tuning using GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf']
}

grid_search = GridSearchCV(estimator=svm_classifier, param_grid=param_grid, cv=2, scoring='accuracy', verbose=2, n_jobs=-1)
grid_search.fit(X_train_resampled, y_train_resampled)

# Use the best estimator from the grid search
best_svm_classifier = grid_search.best_estimator_

# Train the classifier on the resampled training data
best_svm_classifier.fit(X_train_resampled, y_train_resampled)

# Predict on the test set
y_pred = best_svm_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix\n")

# Create a heatmap for the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Ictal', 'Inter-ictal', 'Pre-ictal'],
            yticklabels=['Ictal', 'Inter-ictal', 'Pre-ictal'],
            annot_kws={"size": 16})  # Increase the size of the numbers

# Move the xticklabels to the top
plt.gca().xaxis.set_ticks_position('top')
plt.gca().xaxis.set_label_position('top')

plt.xlabel('Predicted Labels\n', fontsize=14)  # Increase fontsize for x label
plt.ylabel('True Labels', fontsize=14)          # Increase fontsize for y label
plt.title('Confusion Matrix\n', fontsize=16)   # Increase fontsize for title
# Increase font size for tick labels
plt.xticks(fontsize=14)  # Set font size for x-tick labels
plt.yticks(fontsize=14)  # Set font size for y-tick labels
plt.show()

# Define class labels
class_labels = ['Class 0', 'Class 1', 'Class 2']

# Create a DataFrame for the confusion matrix
conf_matrix_df = pd.DataFrame(conf_matrix, index=class_labels, columns=class_labels)

# Print the DataFrame
print("Confusion Matrix:")
print(conf_matrix_df)

print("\nAccuracy:", accuracy * 100)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Define the file path to save the model
model_filename = 'svm_model.pkl'

# Save the trained model to a file
joblib.dump(best_svm_classifier, model_filename)

# Load the saved SVM model
svm_model = joblib.load(model_filename)

def predict_labels(features):
    # Predict labels using the trained model
    predicted_labels = svm_model.predict(features)  # Reshape for single sample prediction
    return predicted_labels

# Assuming 'X' contains the features for which you want to predict labels
predicted_labels = predict_labels(X)  # Use test set features for prediction, not the entire dataset
print("In prediction, the labels represent: Ictal=0, Interictal=1, Preictal=2")
print("Predicted labels:", predicted_labels)
