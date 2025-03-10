import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

def scientific_to_decimal(number, precision=10):
    # Format the number with the given precision
    return f"{number:.{precision}f}"

ictal_periods = []
interictal_periods = []
preictal_periods = []

# Load the EEG data from p11_Record3.edf
raw = mne.io.read_raw_edf('p10_Record1.edf', preload=True)

# Filter parameters
low_freq = 1  # Low cutoff frequency in Hz
high_freq = 70  # High cutoff frequency in Hz

# Apply bandpass filter to remove noise
raw.filter(low_freq, high_freq, fir_design='firwin', l_trans_bandwidth='auto', h_trans_bandwidth='auto',
           filter_length='auto', phase='zero')

# Extract filtered data
data, times = raw[:, :]

# Plot the filtered EEG data for all channels
raw.plot(duration=30, n_channels=len(raw.info['ch_names']), scalings='auto', title='Filtered EEG Signal - All Channels', show=True)
plt.savefig("p10_r1_filtered_eeg_signal.jpg", dpi=300, format='jpg')
plt.show()
# Calculate the average signal across all channels
average_signal = np.mean(data, axis=0)
# Plot the average signal
plt.figure(figsize=(12, 6))
plt.plot(raw.times, average_signal, color='b')
plt.xlabel('Time (s)')
plt.ylabel('Average Signal (µV)')
plt.title('Average Signal Across Channels')

plt.show()
# Detect epochs with high amplitude activity
spike_epochs = []
start_idx = None
max_amplitude = float('-inf')  # Initialize with the smallest possible value

for amplitude in average_signal:
    if amplitude > max_amplitude:
        max_amplitude = amplitude

print(f"Largest amplitude: {max_amplitude}")
# Define thresholds for spike detection
positive_threshold = max_amplitude/2
negative_threshold = -(max_amplitude/2)
preictal_threshold = positive_threshold/2
negative_preictal_threshold = negative_threshold/2
spike_values= []

for i, amplitude in enumerate(average_signal):
    if amplitude > positive_threshold or amplitude < negative_threshold:
        if start_idx is None:
            start_idx = i
    elif start_idx is not None:
        end_idx = i - 1
        spike_epochs.append((start_idx, end_idx))
        spike_values.append(amplitude)
        start_idx = None

# Check if spike epochs were detected
if not spike_epochs:
    print("No spike epochs detected. Adjust the thresholds.")
    exit()
# Create DataFrame
df = pd.DataFrame(
    [(start, end, spike) for (start, end), spike in zip(spike_epochs, spike_values)],
    columns=["Start_Index", "End_Index", "Spike_Amplitude (Hz)"]
)



# Save to CSV
df.to_csv("spike_epochs.csv", index=False)


print("Spike epochs saved to 'spike_epochs.csv'")

# Plot the average signal and highlight detected spikes
plt.figure(figsize=(14, 7))
plt.plot(raw.times, average_signal, label='Average Signal', color='b')

# Highlight detected spike regions
for start_idx, end_idx in spike_epochs:
    plt.axvspan(raw.times[start_idx], raw.times[end_idx], color='red', alpha=0.3, label='Detected Spike' if start_idx == spike_epochs[0][0] else "")

plt.axhline(positive_threshold, color='g', linestyle='--', label='Positive Threshold')
plt.axhline(negative_threshold, color='r', linestyle='--', label='Negative Threshold')
plt.xlabel('Time (s)')
plt.ylabel('Average Signal (µV)')
plt.title('Average Signal Across Channels with Detected Spikes')
plt.legend()
plt.show()
# Find the first preictal spike from the beginning of the average signal
first_preictal_spike_idx = np.where((average_signal > preictal_threshold) |
                                     (average_signal < negative_preictal_threshold))[0][0]

# Reshape the data for clustering (each time point is a feature)
X = np.array(spike_epochs)

# Determine the optimal number of clusters using the elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Find the elbow point
diff = np.diff(wcss)
elbow_point = np.argmin(diff) + 1
num_clusters = elbow_point

# Apply K-means clustering
kmeans = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
pred_y = kmeans.fit_predict(X)

# Group clusters
grouped_clusters = []
group = []
prev_end_time = None
for idx, cluster_label in enumerate(pred_y):
    start_time = raw.times[spike_epochs[idx][0]]
    end_time = raw.times[spike_epochs[idx][1]]
    if prev_end_time is not None and start_time - prev_end_time <= 30:
        group.append(spike_epochs[idx])
    else:
        if group:
            grouped_clusters.append(group)
        group = [spike_epochs[idx]]
    prev_end_time = end_time
if group:
    grouped_clusters.append(group)

# Remove excluded groups
grouped_clusters = [group for group in grouped_clusters if len(group) > 1 and
                    (raw.times[group[-1][1]] - raw.times[group[0][0]]) >= 3]

# Define a colormap
num_groups = len(grouped_clusters)
color_map = plt.cm.get_cmap('tab10', num_groups)

# Plot the average signal
plt.figure(figsize=(12, 6))
plt.plot(raw.times, average_signal, color='b')
plt.xlabel('Time (s)')
plt.ylabel('Average Signal (µV)')
plt.title('Average Signal Across Channels after labelling')

# Plot epochs with shaded areas for each group
for idx, group in enumerate(grouped_clusters):
    start_time = raw.times[group[0][0]]
    end_time = raw.times[group[-1][1]]
    color = color_map(idx % num_groups)
    plt.axvspan(start_time, end_time, color='red', alpha=0.3)
    group_mid_time = (start_time + end_time) / 2
    plt.text(group_mid_time, np.max(average_signal), f'Ictal {idx + 1}', color='black', ha='center')
    ictal_periods.append((start_time, end_time))
    if idx == 0:
        first_group_start_time = start_time
        preictal_periods.append((raw.times[first_preictal_spike_idx], first_group_start_time))

# Plot interictal periods
for i in range(len(grouped_clusters) - 1):
    interictal_start = raw.times[grouped_clusters[i][-1][1]]
    interictal_end = raw.times[grouped_clusters[i + 1][0][0]]
    interictal_color = 'gray'
    plt.axvspan(interictal_start, interictal_end, color=interictal_color, alpha=0.3)
    plt.text(interictal_start + 20, np.max(average_signal), f'Inter-Ictal {i + 1}', color='black',
             ha='center')
    interictal_periods.append((interictal_start, interictal_end))

# Calculate amplitude inside interictal periods and plot if higher than preictal threshold
for period in interictal_periods:
    start_idx = np.where(raw.times == period[0])[0][0]
    end_idx = np.where(raw.times == period[1])[0][0]

    # Exclude the first 10 seconds of each interictal period
    start_idx += int(10 * raw.info['sfreq'])  # Convert 10 seconds to index
    if start_idx >= end_idx:
        continue

    # Calculate amplitude inside the interictal period excluding the start and end points
    interictal_amplitudes = average_signal[start_idx:end_idx]

    # Check if any amplitude values exceed the preictal threshold or negative_preictal_threshold
    exceed_threshold_idx = np.where((interictal_amplitudes > preictal_threshold) |
                                    (interictal_amplitudes < negative_preictal_threshold))[0]
    # Shade the region from the first preictal spike to the start of the first ictal period as preictal
    plt.axvspan(raw.times[first_preictal_spike_idx], first_group_start_time, color='orange', alpha=0.3)

    # Check if any spikes are detected
    if len(exceed_threshold_idx) > 0:
        # Find the first spike in the interictal period
        first_spike_idx = start_idx + exceed_threshold_idx[0]

        # Shade the region from the first spike to the end of the interictal period
        plt.axvspan(raw.times[first_spike_idx], period[1], color='orange', alpha=0.3)

        # Save this shaded region as a preictal period
        preictal_periods.append((raw.times[first_spike_idx], period[1]))

# Label preictal periods on the plot with adjusted positions to avoid overlap
label_positions = []
for period in preictal_periods:
    label_time = (period[0] + period[1]) / 2
    if label_time in label_positions:
        label_time += 0.01  # Adjust the label position slightly
    plt.text(label_time, np.max(average_signal), 'Preictal', color='black', ha='center')
    label_positions.append(label_time)

# Find the end time of the last ictal period
last_ictal_end_time = ictal_periods[-1][1]

# Define the "none" stage period
none_stage_period = (last_ictal_end_time, raw.times[-1])

# Add the "none" stage period to the DataFrame
periods_data = {
    'Start_Time': [period[0] for period in ictal_periods + interictal_periods + preictal_periods + [none_stage_period]],
    'End_Time': [period[1] for period in ictal_periods + interictal_periods + preictal_periods + [none_stage_period]],
    'Amplitude': [average_signal[np.where(raw.times == period[0])[0][0]] for period in ictal_periods + interictal_periods + preictal_periods + [none_stage_period]],
    'Label': ['Ictal'] * len(ictal_periods) + ['Interictal'] * len(interictal_periods) + ['Preictal'] * len(preictal_periods) + ['None']
}

df = pd.DataFrame(periods_data)

# Sort DataFrame by 'Start_Time' in ascending order
df = df.sort_values(by='Start_Time')

# Save to CSV


# Create features for each period
df['Start']=df['Start_Time']
df['end']=df['End_Time']
df['Duration'] = df['End_Time'] - df['Start_Time']
df['Max_Amplitude'] = df.apply(lambda row: np.max(average_signal[np.where(raw.times == row['Start_Time'])[0][0]:np.where(raw.times == row['End_Time'])[0][0]]), axis=1)
df['Min_Amplitude'] = df.apply(lambda row: np.min(average_signal[np.where(raw.times == row['Start_Time'])[0][0]:np.where(raw.times == row['End_Time'])[0][0]]), axis=1)
df['Mean_Amplitude'] = df.apply(lambda row: np.mean(average_signal[np.where(raw.times == row['Start_Time'])[0][0]:np.where(raw.times == row['End_Time'])[0][0]]), axis=1)
df['Std_Amplitude'] = df.apply(lambda row: np.std(average_signal[np.where(raw.times == row['Start_Time'])[0][0]:np.where(raw.times == row['End_Time'])[0][0]]), axis=1)
# Feature 6: Spike Count - Number of spikes within each period
df['Spike_Count'] = df.apply(lambda row: len([1 for epoch in grouped_clusters if row['Start_Time'] <= raw.times[epoch[0][0]] <= row['End_Time']]), axis=1)

# Feature 7: Spike Rate - Average number of spikes per second within each period
df['Spike_Rate'] = df['Spike_Count'] / df['Duration']

# Feature 8: Interictal Duration - Duration since the last ictal period
df['Interictal_Duration'] = df.apply(lambda row: row['Start_Time'] - last_ictal_end_time if row['Label'] == 'Interictal' else 0, axis=1)

# Feature 9: Preictal Duration - Duration until the next ictal period
df['Preictal_Duration'] = df.apply(lambda row: grouped_clusters[0][0][0] - row['End_Time'] if row['Label'] == 'Interictal' and len(grouped_clusters) > 0 else 0, axis=1)

# Feature 10: Change in Spike Rate - Difference in spike rate between consecutive periods
df['Change_in_Spike_Rate'] = df['Spike_Rate'].diff().fillna(0)
# Fill the first row with 0 (since there's no previous period to compare with)
df.loc[0, 'Change_in_Spike_Rate'] = 0


# Encode labels
label_mapping = {'Ictal': 0, 'Interictal': 1, 'Preictal': 2, 'None': 3}
df['Label_Encoded'] = df['Label'].map(label_mapping)

# Split the dataset into training and testing sets
X = df[['Duration', 'Max_Amplitude', 'Min_Amplitude', 'Mean_Amplitude', 'Std_Amplitude','Spike_Count','Spike_Rate','Interictal_Duration','Preictal_Duration','Change_in_Spike_Rate']]
y = df['Label_Encoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
Time=df[['Start','end']]
label=df[['Label_Encoded']]
labelled_Data=pd.concat([Time,label],axis=1)
labelled_Data.to_csv("labelled_Data.csv",index=False)
# Save the prepared dataset to a file
prepared_data = pd.concat([X, y], axis=1)
prepared_data.to_csv('prepared_eeg_data10_1.csv', index=False)

# Print the prepared dataset
print("\nPrepared EEG data:")
print(prepared_data.head())

# Plot the "none" stage on the average signal plot
plt.axvspan(none_stage_period[0], none_stage_period[1], color='gray', alpha=0.3)
plt.text((none_stage_period[0] + none_stage_period[1]) / 2, np.max(average_signal), 'None', color='black', ha='center')

plt.show()



